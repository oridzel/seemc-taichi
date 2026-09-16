# Do not enable ``from __future__ import annotations`` in this module.
# Taichi 1.7.4 needs real dtype objects, not postponed string annotations.


def make_primitives(ti, fp):
    """Return Taichi-callable interpolation and CDF primitives for ``fp``.

    All helpers use a single return at the end.  Taichi 1.7.4 rejects a
    ``return`` nested in runtime control flow inside ``@ti.func``.
    """

    @ti.func
    def lower_bin(grid, n: ti.i32, x: fp) -> ti.i32:
        idx = ti.i32(0)
        if x <= grid[0]:
            idx = ti.i32(0)
        elif x >= grid[n - 1]:
            idx = n - ti.i32(2)
        else:
            lo = ti.i32(0)
            hi = n - ti.i32(1)
            while hi - lo > ti.i32(1):
                mid = (lo + hi) // ti.i32(2)
                if grid[mid] <= x:
                    lo = mid
                else:
                    hi = mid
            idx = lo
        return idx

    @ti.func
    def interp1(grid, values, n: ti.i32, x: fp) -> fp:
        i = lower_bin(grid, n, x)
        dx = grid[i + 1] - grid[i]
        t = fp(0.0)
        if dx > fp(0.0):
            t = (x - grid[i]) / dx
        t = ti.max(fp(0.0), ti.min(fp(1.0), t))
        y = values[i] * (fp(1.0) - t) + values[i + 1] * t
        return y

    @ti.func
    def stochastic_energy_bin(grid, n: ti.i32, x: fp) -> ti.i32:
        i = lower_bin(grid, n, x)
        dx = grid[i + 1] - grid[i]
        t = fp(0.0)
        if dx > fp(0.0):
            t = (x - grid[i]) / dx
        t = ti.max(fp(0.0), ti.min(fp(1.0), t))
        j = i + ti.cast(ti.random(fp) < t, ti.i32)
        return j

    @ti.func
    def inverse_cdf_column(cdf, abscissa, nrow: ti.i32, col: ti.i32, u: fp) -> fp:
        lo = ti.i32(0)
        hi = nrow - ti.i32(1)
        uu = ti.max(fp(0.0), ti.min(fp(1.0), u))
        while hi - lo > ti.i32(1):
            mid = (lo + hi) // ti.i32(2)
            if cdf[mid, col] <= uu:
                lo = mid
            else:
                hi = mid
        c0 = cdf[lo, col]
        c1 = cdf[hi, col]
        a = fp(0.0)
        if c1 > c0:
            a = (uu - c0) / (c1 - c0)
        y = abscissa[lo] * (fp(1.0) - a) + abscissa[hi] * a
        return y

    @ti.func
    def lower_bin_matrix_column(grid, nrow: ti.i32, col: ti.i32, x: fp) -> ti.i32:
        idx = ti.i32(0)
        if x <= grid[0, col]:
            idx = ti.i32(0)
        elif x >= grid[nrow - 1, col]:
            idx = nrow - ti.i32(2)
        else:
            lo = ti.i32(0)
            hi = nrow - ti.i32(1)
            while hi - lo > ti.i32(1):
                mid = (lo + hi) // ti.i32(2)
                if grid[mid, col] <= x:
                    lo = mid
                else:
                    hi = mid
            idx = lo
        return idx

    @ti.func
    def interp_matrix_column(grid, values, nrow: ti.i32, col: ti.i32, x: fp) -> fp:
        i = lower_bin_matrix_column(grid, nrow, col, x)
        x0 = grid[i, col]
        x1 = grid[i + 1, col]
        t = fp(0.0)
        if x1 > x0:
            t = (x - x0) / (x1 - x0)
        t = ti.max(fp(0.0), ti.min(fp(1.0), t))
        y = values[i, col] * (fp(1.0) - t) + values[i + 1, col] * t
        return y

    @ti.func
    def inverse_cdf_matrix_column(
        cdf, abscissa, nrow: ti.i32, col: ti.i32, target: fp
    ) -> fp:
        lo = ti.i32(0)
        hi = nrow - ti.i32(1)
        tt = ti.max(cdf[0, col], ti.min(cdf[nrow - 1, col], target))
        while hi - lo > ti.i32(1):
            mid = (lo + hi) // ti.i32(2)
            if cdf[mid, col] <= tt:
                lo = mid
            else:
                hi = mid
        c0 = cdf[lo, col]
        c1 = cdf[hi, col]
        a = fp(0.0)
        if c1 > c0:
            a = (tt - c0) / (c1 - c0)
        y = abscissa[lo, col] * (fp(1.0) - a) + abscissa[hi, col] * a
        return y

    @ti.func
    def interp2_bilinear(xgrid, ygrid, values, nx: ti.i32, ny: ti.i32, x: fp, y: fp) -> fp:
        ix = lower_bin(xgrid, nx, x)
        iy = lower_bin(ygrid, ny, y)
        x0 = xgrid[ix]
        x1 = xgrid[ix + 1]
        y0 = ygrid[iy]
        y1 = ygrid[iy + 1]
        tx = fp(0.0)
        ty = fp(0.0)
        if x1 > x0:
            tx = (x - x0) / (x1 - x0)
        if y1 > y0:
            ty = (y - y0) / (y1 - y0)
        tx = ti.max(fp(0.0), ti.min(fp(1.0), tx))
        ty = ti.max(fp(0.0), ti.min(fp(1.0), ty))
        v00 = values[ix, iy]
        v10 = values[ix + 1, iy]
        v01 = values[ix, iy + 1]
        v11 = values[ix + 1, iy + 1]
        vx0 = v00 * (fp(1.0) - tx) + v10 * tx
        vx1 = v01 * (fp(1.0) - tx) + v11 * tx
        out = vx0 * (fp(1.0) - ty) + vx1 * ty
        return out

    return (
        lower_bin,
        interp1,
        stochastic_energy_bin,
        inverse_cdf_column,
        lower_bin_matrix_column,
        interp_matrix_column,
        inverse_cdf_matrix_column,
        interp2_bilinear,
    )
