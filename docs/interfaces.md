# Project Interfaces

## 1. Loader Output

The LiDAR loader must return:

- Shape: `(N, 4)`
- Type: NumPy array
- Columns:

| Column | Meaning |
|---|---|
| 0 | X |
| 1 | Y |
| 2 | Z |
| 3 | Intensity / remission |

The loader must preserve the original point order.

---

## 2. Preprocessing Output

Preprocessing receives:

`N × 4` point array

It returns:

- `clean_points`
- `keep_indices`

Required relationship:

`clean_points == original_points[keep_indices]`

Points must not be reordered.

---

## 3. Semantic Output

Semantic processing receives:

- `clean_points`
- `keep_indices`
- original `.label` data

It returns one semantic label for every retained point.

Required relationship:

`len(clean_points) == len(clean_labels)`

Point order and label order must remain aligned.

---

## 4. Adaptive Grid Cell

Each final grid cell must contain:

- `x_min`
- `x_max`
- `y_min`
- `y_max`
- `point_count`
- `z_mean`
- `z_min`
- `z_max`
- `height_range`
- `semantic_class`
- `semantic_confidence`

Semantic class must be obtained using class voting / majority count.

Semantic class IDs must never be averaged numerically.

---

## 5. 2.5D Map

The 2.5D map is a collection of final grid cells.

Each cell represents:

- 2D spatial position and size
- summarized elevation
- summarized semantic information

---

## 6. Current Prototype Scope

The current prototype uses:

- SemanticKITTI ground-truth semantic labels
- Distance-adaptive grid
- Elevation statistics
- Semantic aggregation
- 2.5D map
- Visualization

The current prototype does NOT include:

- neural-network training
- PointNet++
- RangeNet++
- multi-frame persistent mapping
- GPU optimization
- full error-bounded quadtree refinement