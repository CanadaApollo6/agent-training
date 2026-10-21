# `prime pods create --id X` looks the offer up in the unfiltered availability list, which can leave out offers that
# only show up when filtered by GPU type (H200 on Nebius, 2026-10-21). This runs the same command with the lookup
# filtered by the type given in --gpu-type.
#
#   uvx --from prime python prime_create.py --id 1215b4 --gpu-type H200_141GB --name flint-tb2 --disk-size 400 -y --plain
import sys

from prime_cli.api.availability import AvailabilityClient
from prime_cli.main import app

gpu_type = sys.argv[sys.argv.index("--gpu-type") + 1]
get = AvailabilityClient.get
AvailabilityClient.get = lambda self, *a, **k: get(self, *a, **{**k, "gpu_type": k.get("gpu_type") or gpu_type})
sys.argv = ["prime", "pods", "create", *sys.argv[1:]]
app()
