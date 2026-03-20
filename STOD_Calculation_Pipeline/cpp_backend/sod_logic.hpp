#ifndef SOD_LOGIC_HPP
#define SOD_LOGIC_HPP

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <cstdint>

// ---------------------------------------------------------
// Helper: Hashable Grid Point
// ---------------------------------------------------------
struct Point {
    int r, c;
    bool operator==(const Point& other) const {
        return r == other.r && c == other.c;
    }
};

// ---------------------------------------------------------
// Result Structure for STOD
// ---------------------------------------------------------
struct SmodResult {
    int type_code;   // 0 = T, 1 = UC, 2 = UU
    double score;    // Sum of (level * uncanceled_count)
};

// ---------------------------------------------------------
// 1. DISCRETIZE TRAJECTORY
// ---------------------------------------------------------
// NOTE: We do NOT clamp indices to the grid bounds. This allows trajectories
// that leave the analysis domain to have distinct cell visitation patterns,
// which is consistent with how FTLE/FLI/LD handle unbounded trajectories.
// The grid indexing scheme is simply extended beyond the physical domain.
inline std::vector<Point> discretize_path(
    const std::vector<double>& float_path,
    int nx, int ny,
    double x_min, double x_max,
    double y_min, double y_max
) {
    std::vector<Point> grid_path;
    size_t n_points = float_path.size() / 2;
    grid_path.reserve(n_points);

    double dx = x_max - x_min;
    double dy = y_max - y_min;

    for (size_t i = 0; i < n_points; ++i) {
        double x = float_path[2*i];
        double y = float_path[2*i+1];

        // Map to integer indices (NO CLAMPING - allow indices outside [0, nx-1] and [0, ny-1])
        int c = (int)((x - x_min) / dx * nx);
        int r = (int)((y - y_min) / dy * ny);

        grid_path.push_back({r, c});
    }
    return grid_path;
}

// ---------------------------------------------------------
// 2. STOD PAIR LOGIC (Stability Based) - O(n) Optimized
// ---------------------------------------------------------
// Optimization: Instead of checking all previous levels at each step (O(n²)),
// we track which points are "waiting" for a specific row or column to become
// fully covered. When a new row/col is added, we only check the waiting points.
inline SmodResult calculate_stod_pair(
    const std::vector<Point>& path_a, 
    const std::vector<Point>& path_b
) {
    size_t len_a = path_a.size();
    size_t len_b = path_b.size();
    
    // Empty paths -> Terminated (0) with 0 score
    if (len_a == 0 || len_b == 0) return {0, 0.0};

    // Early termination check: Start at same cell
    if (path_a[0] == path_b[0]) {
        return {0, 0.0};
    }

    size_t n = (len_a < len_b) ? len_a : len_b;

    std::unordered_set<int> a_rows, a_cols;
    std::unordered_set<int> b_rows, b_cols;
    a_rows.reserve(n); a_cols.reserve(n);
    b_rows.reserve(n); b_cols.reserve(n);

    // For O(n) termination checking:
    // Track A points waiting for B to cover their row (already have col covered)
    // Track A points waiting for B to cover their col (already have row covered)
    // Same for B points waiting for A
    // Key = row or col value, Value = list of point indices waiting for that value
    std::unordered_map<int, std::vector<size_t>> a_waiting_for_b_row;  // A points with col covered, waiting for row
    std::unordered_map<int, std::vector<size_t>> a_waiting_for_b_col;  // A points with row covered, waiting for col
    std::unordered_map<int, std::vector<size_t>> b_waiting_for_a_row;  // B points with col covered, waiting for row
    std::unordered_map<int, std::vector<size_t>> b_waiting_for_a_col;  // B points with row covered, waiting for col

    bool terminated = false;
    size_t final_level = 0;

    // --- 1. Simulation Phase (O(n) optimized) ---
    for (size_t level = 0; level < n; ++level) {
        final_level = level;
        const Point& pA = path_a[level];
        const Point& pB = path_b[level];

        // Check if newly added B row/col completes any waiting A point
        bool b_row_is_new = (b_rows.find(pB.r) == b_rows.end());
        bool b_col_is_new = (b_cols.find(pB.c) == b_cols.end());
        bool a_row_is_new = (a_rows.find(pA.r) == a_rows.end());
        bool a_col_is_new = (a_cols.find(pA.c) == a_cols.end());

        // Insert into sets
        a_rows.insert(pA.r); a_cols.insert(pA.c);
        b_rows.insert(pB.r); b_cols.insert(pB.c);

        // Check if new B row completes any A point waiting for that row
        if (b_row_is_new && a_waiting_for_b_row.count(pB.r)) {
            terminated = true;
        }
        // Check if new B col completes any A point waiting for that col
        if (!terminated && b_col_is_new && a_waiting_for_b_col.count(pB.c)) {
            terminated = true;
        }
        // Check if new A row completes any B point waiting for that row
        if (!terminated && a_row_is_new && b_waiting_for_a_row.count(pA.r)) {
            terminated = true;
        }
        // Check if new A col completes any B point waiting for that col
        if (!terminated && a_col_is_new && b_waiting_for_a_col.count(pA.c)) {
            terminated = true;
        }

        if (terminated) {
            break;
        }

        // Register current A point's waiting status
        bool a_row_covered = (b_rows.count(pA.r) > 0);
        bool a_col_covered = (b_cols.count(pA.c) > 0);
        if (a_row_covered && a_col_covered) {
            // Already fully covered - terminate
            terminated = true;
            break;
        } else if (a_row_covered) {
            // Row covered, waiting for col
            a_waiting_for_b_col[pA.c].push_back(level);
        } else if (a_col_covered) {
            // Col covered, waiting for row
            a_waiting_for_b_row[pA.r].push_back(level);
        }
        // If neither covered, this point can't trigger termination (needs both)

        // Register current B point's waiting status
        bool b_row_covered = (a_rows.count(pB.r) > 0);
        bool b_col_covered = (a_cols.count(pB.c) > 0);
        if (b_row_covered && b_col_covered) {
            // Already fully covered - terminate
            terminated = true;
            break;
        } else if (b_row_covered) {
            // Row covered, waiting for col
            b_waiting_for_a_col[pB.c].push_back(level);
        } else if (b_col_covered) {
            // Col covered, waiting for row
            b_waiting_for_a_row[pB.r].push_back(level);
        }
    }

    // --- 2. Scoring Phase ---
    double total_score = 0.0;
    bool has_any_cancellation = false;

    for (size_t level = 0; level <= final_level; ++level) {
        const Point& pA = path_a[level];
        const Point& pB = path_b[level];

        int canc = 0;
        if (b_rows.count(pA.r)) canc++;
        if (b_cols.count(pA.c)) canc++;
        if (a_rows.count(pB.r)) canc++;
        if (a_cols.count(pB.c)) canc++;

        if (canc > 0) has_any_cancellation = true;

        // Score is based on UNCANCELED count
        int unc = 4 - canc;
        total_score += (double)(level * unc);
    }

    // --- 3. Determine Type ---
    int type_code = 0; // Default T
    if (terminated) {
        type_code = 0; // Type T
    } else if (has_any_cancellation) {
        type_code = 1; // Type UC
    } else {
        type_code = 2; // Type UU
    }

    return {type_code, total_score};
}

#endif

