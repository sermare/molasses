"""Static checks on the Slurm templates: every GPU job is single-GPU, low priority, requeueable, and keeps off the known-bad nodes."""
import re
import pytest

GPU = ["pass1_embed.sbatch", "pass2_tail.sbatch", "train_score_seed.sbatch", "score_base.sbatch", "score_chunk.sbatch"]


@pytest.mark.parametrize("name", GPU)
def test_gpu_template_requests(repo, name):
    t = (repo / "slurm" / name).read_text()
    assert re.search(r"#SBATCH --gres=gpu:1\b", t) and "--nodes=1" in t
    assert "--qos=savio_lowprio" in t and "--requeue" in t
    assert "n0176" in t, "the bad node n0176 must stay excluded"
    assert re.search(r"#SBATCH --time=\d+:\d\d:\d\d", t)


def test_pass1_task_requeues_itself_when_preempted(repo):
    t = (repo / "slurm" / "pass1_embed.sbatch").read_text()
    assert "scontrol requeue" in t and "SLURM_RESTART_COUNT" in t        # pre-emption makes boltz exit 0; without this the work is lost
