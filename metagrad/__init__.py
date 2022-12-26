from metagrad.tensor import Tensor # 必须在第一行，先执行_register_ops
import metagrad.functions
import metagrad.init
import metagrad.ops
from metagrad import cuda

ops.install_ops()
