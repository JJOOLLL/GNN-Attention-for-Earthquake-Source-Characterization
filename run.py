"""Display configuration by default; --execute explicitly starts full training."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ['transformer_attention','transformer_max','no_message_attention','no_message_max','gcn_attention','gcn_max']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--region', choices=['California','Alaska'])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--save-root', type=Path)
    parser.add_argument('--kernel', default='gnn-paper')
    parser.add_argument('--graph', choices=['knn', 'median'],
                        help='GCN graph only (default: knn); other models keep their original graph')
    parser.add_argument('--resume', action='store_true', help='Resume a verified GCN checkpoint')
    parser.add_argument('--strict-reference', action='store_true',
                        help='GCN only: require original-environment initialization/RNG digests')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not args.model or not args.region:
        if args.execute:
            parser.error('--execute requires --model and --region')
        print('\n'.join(MODELS))
        print('Regions: California, Alaska. No training started.')
        return
    settings = json.loads((ROOT/'experiments/settings.json').read_text(encoding='utf-8'))
    is_gcn = args.model in ('gcn_max', 'gcn_attention')
    if not is_gcn and (args.graph is not None or args.resume or args.strict_reference):
        parser.error('--graph, --resume and --strict-reference apply only to GCN models')
    graph = args.graph or 'knn'
    selected = dict(settings[f'{args.region}/{args.model}'])
    if is_gcn:
        selected.update(selected_graph=graph, resume_requested=args.resume,
                        strict_reference=args.strict_reference)
        selected['message_passing'] = selected['graph_definitions'][graph]
        selected['checkpoint'] = selected['checkpoint'].format(graph=graph)
    print(json.dumps(selected, ensure_ascii=False, indent=2))
    if not args.execute:
        print('Configuration only. Add --execute to start full training.')
        return
    if args.data_root:
        os.environ['GNN_DATA_ROOT'] = str(args.data_root.resolve())
    if args.save_root:
        os.environ['GNN_SAVE_ROOT'] = str(args.save_root.resolve())
    from experiments.paths import get_data_root, get_save_root, check_input_hashes
    data = get_data_root()/('data_DA' if args.region=='California' else 'data_ANCHORAGE_DA')
    check_input_hashes(args.region, data)
    if is_gcn:
        save = get_save_root()/args.region/args.model/graph
        if args.resume:
            if not (save/'preflight.json').is_file():
                raise FileNotFoundError(f'No compatible GCN run to resume: {save}')
        elif save.exists():
            raise FileExistsError(f'Existing GCN run directory; use a new save root or --resume: {save}')
        from training.gcn_runner import train
        train(args.model, args.region, save, resume=args.resume,
              graph_scheme=graph, strict_reference=args.strict_reference)
        return
    save = get_save_root()/args.region/args.model
    if any((save/n).exists() for n in ['best-model.pth','last-model.pth']):
        raise FileExistsError(f'Existing checkpoint directory: {save}')
    import nbformat
    from nbclient import NotebookClient
    source = ROOT/'experiments'/args.region/(args.model+'.ipynb')
    output = ROOT/'runs'/args.region/(args.model+'.ipynb')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(source, as_version=4)
    try:
        NotebookClient(nb, timeout=None, kernel_name=args.kernel,
                       resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        nbformat.write(nb, output)


if __name__=='__main__':
    main()
