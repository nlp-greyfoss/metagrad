from metagrad.tensor import Tensor # 必须在第一行，先执行_register_ops
import metagrad.functions
import metagrad.init
import metagrad.ops
from metagrad import cuda

ops.install_ops()

from metagrad.tensor import no_grad
from metagrad.tensor import ensure_tensor
from metagrad.tensor import ensure_array
from metagrad.tensor import float_type
from metagrad.tensor import debug_mode

from metagrad import module as nn
from metagrad import optim