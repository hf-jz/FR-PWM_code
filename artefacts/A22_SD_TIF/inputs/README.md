# Real data used by case F (E10)

Ten files, `level2_<shot>.npz`, for shots 28315--28324: the calibrated charge-exchange products of the public
MAST level-2 release,

    https://s3.echo.stfc.ac.uk/mast/level2/shots/<shot>.zarr      (group: charge_exchange)

Variables: `major_radius` (161), `time`, `t_i` (`[161, n_time]`, eV), `v_i` (same shape, m/s).
Fetched with `tools/fetch_mast_level2.py`, which reads the zarr v3 objects directly (the installed zarr is 2.x)
and writes exactly what the store holds: no interpolation, no filling, NaN stays NaN.

The analysis uses, per shot, the frame that carries the most usable radii above R = 0.75 m (twelve radii), not a
fixed time: the campaign does not share a flat-top window, and three shots have no CXRS data at 0.125 s at all.
Nothing in the data is simulated; the only model is the two-parameter exponential profile used to localise the
analysis, and its worst-node residual against the data is reported per shot (1.3 to 4.1 sigma) so the fit can be
checked before the grades are read.

The noise entering the analysis is *not* an instrument error bar, which this release does not carry: it is the
measured radial roughness of the profile, sqrt(pi/6) times the RMS second difference, 6.8 to 35.9 per cent across
the campaign. The structural grades (Gamma) do not depend on it; the quoted sigmas do, and every one of them
scales linearly with it, which is why the noise multiplier at which the grade is lost is reported beside them.

Shot 28321 is the only one whose certificate is refused at the measured noise (35.9 per cent, worst-node fit
4.1 sigma): that refusal is the state boundary reported in the paper, not a missing result.
