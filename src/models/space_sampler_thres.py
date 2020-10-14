"""Spatial sampler with threshold
"""
import torch
import numpy as np
from skimage import measure
from scipy.optimize import linear_sum_assignment


class SpatialSamplerThres():
    def __init__(self, top_k, min_b_size, max_b_size, alpha, beta, img_size):
        """Initialize the spatial sampler.

        Mask will be created from attention by thresholding as:
            mask = alpha * attn.mean() + beta * attn.std()

        Args:
            top_k: (int) number of top regions to sample
            min_b_size: (int) minimum size of each bbox to sample
            max_b_size: (int) maximum size of each bbox to sample
            alpha: (float) parameter to threshold the attention
            beta: (float) parameter to threshold the attention
            img_size: (int) original image size
        """
        # super(SpatialSampler, self).__init__(device)

        self.top_k = top_k
        self.min_b_size = min_b_size
        self.max_b_size = max_b_size
        self.alpha = alpha
        self.beta = beta
        self.img_size = img_size

        self._prev_bboxes = None

    def reset(self):
        """Reset _prev_bbox
        """
        self._prev_bboxes = None

    def sample_frame(self, x, attn, reorder):
        """Sampling function

        Args:
            x: batch of current frames. Shape of (B, C1, H1, W1)
            attn: attention of the current frame. Shape of (B, C2, H2, W2)
                Can be groundtruth attention or halllucination from prev frame

        Return:
            results: tensor of shape [B, top_k, 4]. The last dimension defines
                the bounding boxes as (top, left, bottom, right)
        """
        assert x.shape[0] == attn.shape[0]
        assert x.shape[-1] == x.shape[-2]
        assert attn.shape[-1] == attn.shape[-2]
        batch_size = x.shape[0]
        if self._prev_bboxes is not None:
            assert len(self._prev_bboxes) == batch_size

        # Flatten the attention
        attn = attn.mean(dim=1)

        # For each sample in batch
        results = []
        for b in range(batch_size):
            # Get bboxes in attention plane
            props, top_segids = self._get_bbox_from_attn(
                attn[b].cpu().detach().numpy(), simple_return=True)

            # Project to image plane
            bboxes = []
            for sid in top_segids:
                top, left, bottom, right = self._project_image_plane(
                    props[sid], attn_size=attn.shape[-1])
                bboxes.append([top, left, bottom, right])

            # Append the last item if not enough bboxes
            if len(bboxes) != self.top_k:
                bboxes += [bboxes[-1]] * (self.top_k - len(bboxes))

            # Reorder
            if reorder:
                if (self._prev_bboxes is not None) and (self._prev_bboxes[b] is not None):
                    bboxes = self._sort_bboxes(self._prev_bboxes[b], bboxes)
                else:
                    if self._prev_bboxes is None:
                        self._prev_bboxes = [None for _ in range(batch_size)]
                    self._prev_bboxes[b] = bboxes

            # Collect results
            results.append(bboxes)
        return np.array(results)

    def _get_bbox_from_attn(self, attn, simple_return=True):
        """Segment and get bbox from attention map

        Args:
            attn: attention map
            simple_return: (boolean) If True, return only props and top_segids.
                Otherwise, return also mask, segments, and scores

        Return:
            props: list of RegionProperties
            top_segids: list of segment id of the top segments based on scores
            mask: binary mask of the attention map after thresholding
            segments: labeled segments generated from mask
            scores: array of scores wrt each segment
        """
        # Mask the attention by thresholding
        thres = self.alpha*attn.mean() + self.beta*attn.std()
        mask = np.where(attn > thres, 1.0, 0.0)

        # Segment the mask, each segment will be assigned a label
        segments, n_seg = measure.label(mask, return_num=True)

        # Find bounding boxes
        props = measure.regionprops(segments)

        # Find the top segments
        scores = np.zeros(n_seg)
        for i, prop in enumerate(props):
            scores[i] = attn[prop.coords[:, 0], prop.coords[:, 1]].sum()
        top_segids = scores.argsort()[::-1][:self.top_k]

        if simple_return:
            return props, top_segids
        return props, top_segids, mask, segments, scores

    def _project_image_plane(self, prop, attn_size):
        """Project a bounding box to image plane

        Args:
            prop: (RegionProperties) property of a region

        Return:
            top, left, bottom, right: corner positions of the bbox in image plane
        """
        # Get the scale from original image size to the current attention size
        img_size = self.img_size
        scale = img_size / attn_size

        # Get the square bbox in attn plane
        min_row, min_col, max_row, max_col = prop.bbox
        b_h, b_w = max_row-min_row, max_col-min_col
        b_size = max(b_h, b_w)
        b_center = ((max_row+min_row)//2, (max_col+min_col)//2)

        # Convert to the scale of image plane
        b_center_img = (int(b_center[0]*scale), int(b_center[1]*scale))
        b_size_img = np.clip(int(b_size*scale), self.min_b_size, self.max_b_size)

        top = b_center_img[0] - b_size_img//2
        left = b_center_img[1] - b_size_img//2
        bottom = b_center_img[0] + b_size_img//2
        right = b_center_img[1] + b_size_img//2

        # Adjust the bbox to get a square one
        if top < 0:
            bottom -= top
            top -= top
        if left < 0:
            right -= left
            left -= left
        if bottom > img_size:
            top -= (bottom - img_size)
            bottom -= (bottom - img_size)
        if right > img_size:
            left -= (right - img_size)
            right -= (right - img_size)

        assert (bottom-top == b_size_img) and (right-left == b_size_img), \
            'Bounding box out of bound: {}, {}, {}, {}'.format(
                top, left, bottom, right)

        return top, left, bottom, right

    def _sort_bboxes(self, prev_bboxes, bboxes):
        """Sort bboxes by the min distance wrt the previous bboxes using
        Hungarian algorithm. Each bbox is represented as a tuple of 4 numbers:
        (top, left, bottom, right)

        Args:
            prev_bboxes: list of bboxes in the previous frame
            bboxes: list of bboxes in the current frame

        Return:
            Sorted list of bboxes
        """
        # Find the centers of the bboxes
        prev_centers = [((t+b)//2, (l+r)//2) for t, l, b, r in prev_bboxes]
        centers = [((t+b)//2, (l+r)//2) for t, l, b, r in bboxes]
        M, N = len(prev_centers), len(centers)
        assert M == N, 'Number of bboxes does not match: {}, {}'.format(M, N)

        # Compute cost matrix as squared L2 distances between all center pairs
        cost = np.zeros((M, N), dtype=np.float32)
        for i in range(M):
            for j in range(N):
                p1 = prev_centers[i]
                p2 = centers[j]
                cost[i, j] = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

        # Matching using Hungarian algorithm
        _, col_ind = linear_sum_assignment(cost)

        return [bboxes[i] for i in col_ind]


if __name__ == '__main__':
    from time import time

    spatial_sampler = SpatialSamplerThres(
        top_k=3, min_b_size=64, max_b_size=112, alpha=1.0, beta=0.1, img_size=224)

    batch = 5
    length = 10
    x = torch.rand((batch, length, 3, 224, 224), dtype=torch.float32).cuda()
    attn = torch.rand([batch, length, 64, 14, 14], dtype=torch.float32).cuda()

    spatial_sampler.reset()
    for t in range(length):
        st = time()
        results = spatial_sampler.sample_frame(x[:, t], attn[:, t], reorder=True)
        assert results.shape == (batch, 3, 4)
        print(time() - st)
