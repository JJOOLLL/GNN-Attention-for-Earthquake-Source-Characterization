"""Check bundled predictions without loading models, waveforms, or checkpoints."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    metrics = pd.read_csv(ROOT/'results/metrics.csv', float_precision='round_trip')
    assert len(metrics)==64 and not metrics.duplicated(['model','region','variable']).any()
    names = ['Latitude','Longitude','Depth','Magnitude']
    cols = ['latitude_deg','longitude_deg','depth_km','magnitude']
    keys = ['MAE','MAE_std','MSE','MSE_std','R2']
    error = dict.fromkeys(keys, 0.0)
    count = 0
    for region in ['California','Alaska']:
        sets = {s:set(pd.read_csv(ROOT/'results/splits'/region/(s+'.csv')).event_id)
                for s in ['train','validation','test']}
        assert not (sets['train'] & sets['validation'] or sets['train'] & sets['test'] or sets['validation'] & sets['test'])
        for path in sorted((ROOT/'results/predictions'/region).glob('*.csv')):
            frame = pd.read_csv(path, float_precision='round_trip')
            assert frame.event_id.is_unique and set(frame.event_id)==sets['test']
            assert np.isfinite(frame.to_numpy()).all()
            p = np.ascontiguousarray(frame[['predicted_'+c for c in cols]].to_numpy(dtype=np.float32))
            y = np.ascontiguousarray(frame[['true_'+c for c in cols]].to_numpy(dtype=np.float32))
            absolute = np.abs(p-y)
            square = (p-y)**2
            for i,k in enumerate([110,92] if region=='California' else [111,54]):
                absolute[:,i] *= k
                square[:,i] *= k*k
            r2 = np.array([1-np.sum((y[:,i]-p[:,i])**2)/np.sum((y[:,i]-y[:,i].mean())**2) for i in range(4)])
            values = dict(MAE=absolute.mean(0),MAE_std=absolute.std(0),MSE=square.mean(0),MSE_std=square.std(0),R2=r2)
            for i,v in enumerate(names):
                row = metrics.loc[(metrics.model==path.stem)&(metrics.region==region)&(metrics.variable==v)].iloc[0]
                for key in keys:
                    delta = abs(float(values[key][i])-float(row[key]))
                    error[key] = max(error[key],delta)
                    assert delta <= 1e-6, (path.name,v,key,delta)
            count += 1
    assert count==12
    report = {'status':'passed','own_model_prediction_files':count,'own_model_metric_rows_checked':48,
              'baseline_metric_rows_retained_without_source_code':16,'max_absolute_error':error,
              'training_performed':False}
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    main()
