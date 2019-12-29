import numpy as np


class PriorBoxes:
    def __init__(self, strides, scales, ratios):
        self.strides = strides
        self.scales = scales  # [10, 25, 40]
        self.ratios = ratios
        self.config = {
            "strides": self.strides,
            "scales": self.scales,
            "ratios": self.ratios
        }
        """

        example)
        strides = [4, 8, 16]
        scales => [10, 25, 40] 
        ratios => [(1  ,1),
                   (1.5,0.5),
                   (1.2,0.8),
                   (0.8,1.2),
                   (1.4,1.4)]
        """
    def generate(self, image_shape):
        """
        image_shape(H,W,3)에 맞춰서, Prior Box(==Default Boxes)를 생성하는 코드

        return :
        (# Prior Boxes, 4)로 이루어진 출력 값 생성
        """
        fmap_hs = np.ceil(image_shape[0] / np.asarray(self.strides))
        fmap_ws = np.ceil(image_shape[1] / np.asarray(self.strides))
        total_anchors = []

        # scaled_ratios
        self.ratios = np.asarray(self.ratios, dtype=np.float32)
        scaled_ratios = []
        for s in self.scales:
            for r in self.ratios:
                scaled_ratios.append(s * r)

        scaled_ratios = np.asarray(scaled_ratios)
        scaled_ratios = scaled_ratios.reshape([len(self.scales), len(self.ratios), 2])

        for ind in range(len(self.scales)):
            h = fmap_hs[ind]
            w = fmap_ws[ind]
            stride = self.strides[ind]  # shape []
            achr_sizes = scaled_ratios[ind]
            n_achr_sizes = len(achr_sizes)
            n_achr = (h * w * n_achr_sizes).astype(np.int32)

            # cx
            cx, cy = np.meshgrid(np.arange(w), np.arange(h))
            cx = cx * stride + stride // 2  # shape 32,32,5
            grid_cx = np.stack([cx] * n_achr_sizes, axis=-1)  # shape: (32,32,5)

            # cy
            cy = cy * stride + stride // 2  # shape 32,32,5
            grid_cy = np.stack([cy] * n_achr_sizes, axis=-1)  # shape: (32,32,5)

            #
            grid = np.expand_dims(np.ones_like(cx), axis=-1)  # shape: (32,32, 1)

            # ws
            ws_sizes = achr_sizes[:, 1]  # shape: 5,
            grid_ws = grid * ws_sizes  # shape: (32, 32, 5)

            # hs
            hs_sizes = achr_sizes[:, 0]  # shape: 5,
            grid_hs = grid * hs_sizes  # shape: (32,32, 5)

            # concatenate cx, cy, ws, hw
            anchors = np.stack([grid_cx, grid_cy, grid_ws, grid_hs], axis=-1)  # shape: (32,32,5,4)

            anchors = anchors.reshape([n_achr, 4])  # shape: (32,)
            total_anchors.append(anchors)


        total_anchors = np.concatenate(total_anchors, axis=0)
        return total_anchors
