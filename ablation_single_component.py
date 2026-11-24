"""Task 5: cumulative ablations, one component per step.

python ablation_single_component.py --data-folder public_test

The script changes only student functions in memory. The original generator,
runner and scoring functions are used without modifications. hw4.py is not edited.
"""
from __future__ import annotations
import argparse
import ast
import csv
import json
import shutil
from pathlib import Path
import subprocess
import sys
import types

CHANGES = {
    'B0': 'Naive get_text + recursive splitting + whitespace BM25 top-5',
    'V1': 'Add table-to-text conversion',
    'V2': 'Replace recursive splitting with section/table-aware splitting',
    'V3': 'Add contextual chunk headers, including research broker and report title',
    'V4': 'Add company/type/year metadata filtering',
    'V5': 'Replace whitespace tokenisation with normalised tokenisation',
    'V6': 'Add query rewriting: remove entity/year terms and expand synonyms',
    'V7': 'Add table score boost',
    'V8': 'Reserve an evidence slot for each comparison company',
    'V9': 'Add LLM reranking',
    'V10': 'Apply the reranker answerability decision',
}


class VariantTransformer(ast.NodeTransformer):
    """Separate the combined switches in the supplied hw4.py implementation."""
    def __init__(self):
        self.scope = []
        self.parents = []
        self.edits = 0
        self.header_edits = 0

    def visit(self, node):
        self.parents.append(node)
        result = super().visit(node)
        self.parents.pop()
        return result

    def visit_FunctionDef(self, node):
        self.scope.append(node.name)
        node = self.generic_visit(node)
        if self.scope == ['chunk_document', 'emit']:
            for pos, statement in enumerate(node.body):
                if (isinstance(statement, ast.For)
                    and isinstance(statement.iter, ast.Call)
                    and isinstance(statement.iter.func, ast.Name)
                    and statement.iter.func.id == '_split'):
                    reset = ast.parse('if _ABL_STAGE < 3:\n    prefix = ""').body[0]
                    node.body.insert(pos, reset)
                    self.header_edits += 1
                    break
        self.scope.pop()
        return node

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if not (isinstance(node.left, ast.Name) and node.left.id == '_LEVEL'
                and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant)):
            return node
        value = node.comparators[0].value
        root = self.scope[0] if self.scope else ''
        replacement = None
        if root == 'chunk_document':
            replacement = value
        elif root == '_tokens':
            replacement = 5
        elif root == 'retrieve':
            if value == 4:
                # The table boost is guarded by a Boolean expression.
                replacement = 7 if isinstance(self.parents[-2], ast.BoolOp) else 6
            else:
                replacement = {3: 4, 5: 8, 6: 9, 7: 10}.get(value)
        if replacement is not None:
            node.left.id = '_ABL_STAGE'
            node.comparators[0].value = replacement
            self.edits += 1
        return node


def load_variant(path, variant):
    path = Path(path).resolve()
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    transformer = VariantTransformer()
    tree = transformer.visit(tree)
    if transformer.header_edits != 1 or transformer.edits < 8:
        raise RuntimeError('Unsupported hw4.py structure. Use the complete hw4.py supplied in this chat.')
    ast.fix_missing_locations(tree)
    module = types.ModuleType('_hw4_ablation')
    module.__file__ = str(path)
    module._ABL_STAGE = list(CHANGES).index(variant)
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-folder', default='public_test')
    parser.add_argument('--output-dir', default='ablation_single_results')
    parser.add_argument('--variants', nargs='+', choices=list(CHANGES), default=list(CHANGES))
    parser.add_argument('--worker', choices=list(CHANGES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data = Path(args.data_folder).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        module = load_variant(root / 'hw4.py', args.worker)
        sys.argv = ['hw4.py', '--data-folder', str(data),
                    '--output', str(output / f'{args.worker}_results.csv')]
        module.main()
        return

    gold = json.loads((data / 'ground_truth.json').read_text(encoding='utf-8'))
    summary, errors = [], []
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    for variant in args.variants:
        print(f'Running {variant}: {CHANGES[variant]}', flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', variant,
                   '--data-folder', str(data), '--output-dir', str(output)]
        with (output / f'{variant}.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=1800)
        with (output / f'{variant}_results.csv').open(encoding='utf-8', newline='') as stream:
            rows = [r for r in csv.DictReader(stream) if r['id'] in gold]
        yes = [r for r in rows if gold[r['id']]['evidence']]
        no = [r for r in rows if not gold[r['id']]['evidence']]
        blank = [r for r in yes if not r['answer'].strip()]
        if blank:
            raise RuntimeError(f'{variant}: {len(blank)} answerable questions have empty answers. '
                               f'Check {variant}.log for API errors before reporting this experiment.')
        summary.append(dict(variant=variant, change=CHANGES[variant],
            recall_at_5=mean([r['retrieval_hit'] == 'True' for r in yes]),
            kept=mean([bool(r['retrieved']) for r in yes]),
            rejected=mean([not r['retrieved'] for r in no]),
            accuracy=mean([r['correct'] == 'True' for r in rows])))
        for row in rows:
            if row['correct'] != 'True':
                errors.append(dict(variant=variant, id=row['id'], category=row['category'],
                    question=row['question'], answer=row['answer'], expected=gold[row['id']]['answer'],
                    retrieval_hit=row['retrieval_hit'], retrieved=row['retrieved']))

    with (output / 'summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (output / 'errors.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['variant', 'id', 'category', 'question',
                               'answer', 'expected', 'retrieval_hit', 'retrieved'])
        writer.writeheader()
        writer.writerows(errors)
    lines = ['| Variant | Only change from previous row | Recall@5 | Kept | Rejected | Accuracy |',
             '|---|---|---:|---:|---:|---:|']
    for row in summary:
        cells = [row['variant'], row['change']] + [f"{row[k]:.1%}" for k in
                 ('recall_at_5', 'kept', 'rejected', 'accuracy')]
        lines.append('| ' + ' | '.join(cells) + ' |')
    (output / 'table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if 'V10' in args.variants:
        shutil.copyfile(output / 'V10_results.csv', root / 'results.csv')
        print('Copied the exact final-configuration results to results.csv.')
    print('\n'.join(lines))
    print(f'Completed. Copy table.md into README.md; use errors.csv for error analysis. Outputs: {output}')


if __name__ == '__main__':
    main()
