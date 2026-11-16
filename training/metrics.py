"""Physical-unit metrics preserved from the final experiment implementation."""
def metrics(y, p, region):
    import numpy as np
    from sklearn.metrics import r2_score
    y, p = y.astype(np.float64), p.astype(np.float64)
    scale = np.array([110, 92, 1, 1] if region == 'California' else [111, 54, 1, 1])
    err = (p-y) * scale
    a, b = np.deg2rad(y[:, :2]), np.deg2rad(p[:, :2])
    delta = b-a
    h = np.sin(delta[:, 0]/2)**2 + np.cos(a[:, 0])*np.cos(b[:, 0])*np.sin(delta[:, 1]/2)**2
    dist = 2*6371.0088*np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    return dict(events=len(y), target_order=['latitude','longitude','depth','magnitude'],
        mae_units=['km','km','km','magnitude'], MAE=np.abs(err).mean(0).tolist(),
        MAE_std=np.abs(err).std(0).tolist(), MSE=(err**2).mean(0).tolist(),
        MSE_std=(err**2).std(0).tolist(), R2=[r2_score(y[:, i],p[:, i]) for i in range(4)],
        epicentral_mean_km=float(dist.mean()), epicentral_median_km=float(np.median(dist)),
        epicentral_p90_km=float(np.percentile(dist,90)), within20km=int((dist<=20).sum()),
        std_definition='population spread across events, not random seeds')
