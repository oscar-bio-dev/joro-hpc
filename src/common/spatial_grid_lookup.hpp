#pragma once

#include "common/types.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace joro {

/// Índice espacial 2D O(1) para mapeo ultrarrápido de coordenadas (lon, lat) a cell_id
class SpatialGridLookup {
public:
    void build(const std::vector<GridCell>& cells) {
        if (cells.empty()) return;

        min_lon_ = 1e9f; max_lon_ = -1e9f;
        min_lat_ = 1e9f; max_lat_ = -1e9f;
        for (const auto& c : cells) {
            min_lon_ = std::min(min_lon_, c.centroid_lon);
            max_lon_ = std::max(max_lon_, c.centroid_lon);
            min_lat_ = std::min(min_lat_, c.centroid_lat);
            max_lat_ = std::max(max_lat_, c.centroid_lat);
        }

        // Margen de seguridad exterior
        min_lon_ -= 0.5f; max_lon_ += 0.5f;
        min_lat_ -= 0.5f; max_lat_ += 0.5f;

        // Tamaño de celda espacial ~0.045 grados (~5 km)
        cell_size_deg_ = 0.045f;
        inv_cell_size_ = 1.0f / cell_size_deg_;

        n_cols_ = static_cast<int>(std::ceil((max_lon_ - min_lon_) * inv_cell_size_)) + 1;
        n_rows_ = static_cast<int>(std::ceil((max_lat_ - min_lat_) * inv_cell_size_)) + 1;

        grid_.assign(static_cast<size_t>(n_cols_) * n_rows_, -1);

        for (const auto& c : cells) {
            int col = static_cast<int>((c.centroid_lon - min_lon_) * inv_cell_size_);
            int row = static_cast<int>((c.centroid_lat - min_lat_) * inv_cell_size_);
            if (col >= 0 && col < n_cols_ && row >= 0 && row < n_rows_) {
                grid_[static_cast<size_t>(row) * n_cols_ + col] = c.cell_id;
            }
        }
    }

    [[nodiscard]] int32_t find_cell(float lon, float lat) const noexcept {
        if (grid_.empty()) return -1;

        int base_col = static_cast<int>((lon - min_lon_) * inv_cell_size_);
        int base_row = static_cast<int>((lat - min_lat_) * inv_cell_size_);

        // 1. Verificación directa O(1) en el bin exacto
        if (base_col >= 0 && base_col < n_cols_ && base_row >= 0 && base_row < n_rows_) {
            int32_t cid = grid_[static_cast<size_t>(base_row) * n_cols_ + base_col];
            if (cid != -1) return cid;
        }

        // 2. Verificación en vecindario contiguo 3x3 (máximo 8 vecinos en memoria contigua)
        for (int dr = -1; dr <= 1; ++dr) {
            int r = base_row + dr;
            if (r < 0 || r >= n_rows_) continue;
            for (int dc = -1; dc <= 1; ++dc) {
                int c = base_col + dc;
                if (c < 0 || c >= n_cols_) continue;
                int32_t cid = grid_[static_cast<size_t>(r) * n_cols_ + c];
                if (cid != -1) return cid;
            }
        }

        return -1;
    }

    [[nodiscard]] bool empty() const noexcept { return grid_.empty(); }

private:
    float min_lon_ = 0.0f;
    float max_lon_ = 0.0f;
    float min_lat_ = 0.0f;
    float max_lat_ = 0.0f;
    float cell_size_deg_ = 0.045f;
    float inv_cell_size_ = 1.0f / 0.045f;
    int n_cols_ = 0;
    int n_rows_ = 0;
    std::vector<int32_t> grid_;
};

} // namespace joro
