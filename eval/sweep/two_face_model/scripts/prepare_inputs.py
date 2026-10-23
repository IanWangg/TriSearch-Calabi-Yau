"""Export the training splitter's held-out polytopes for a paired model study."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from core.cy_data_utils import split_rows_by_vertex_count
from mdp.cy_rollout import load_cy_sample_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--num_eval_polytopes", type=int, default=100)
    parser.add_argument("--num_polytopes", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, default=Path("eval/sweep/two_face_model/data"))
    args = parser.parse_args()
    if args.num_polytopes <= 0:
        parser.error("num_polytopes must be positive.")
    source = Path(args.dataset_path).expanduser().resolve()
    split = split_rows_by_vertex_count(load_cy_sample_rows(str(source)), num_eval_polytopes=args.num_eval_polytopes)
    indices = split.eval_polytope_indices[:args.num_polytopes]
    if len(indices) != args.num_polytopes:
        parser.error("Not enough held-out polytopes for this study.")
    rows_by_index = {int(row["polytope_index"]): row for row in split.eval_rows}
    selected = [rows_by_index[index] for index in indices]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "held_out_polytopes.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8")
    metadata = dict(dataset_path=str(source), dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    splitter="split_rows_by_vertex_count", num_eval_polytopes=args.num_eval_polytopes,
                    train_polytope_indices=split.train_polytope_indices,
                    eval_polytope_indices=split.eval_polytope_indices, selected_polytope_indices=indices,
                    output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    (args.output_dir / "split.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
