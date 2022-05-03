import numpy

gpu_available = True


class Device:
    '''
    自定义的设备类
    '''

    @property
    def xp(self):
        '''返回cupy或numpy'''
        raise NotImplementedError("Can not call from base class")

    @property
    def name(self):
        '''设备名称'''
        raise NotImplementedError("Can not call from base class")

    def transfer(self, array):
        '''将array转移到此设备'''
        raise NotImplementedError("Can not call from base class")

    def __eq__(self, other):
        raise NotImplementedError("Can not call from base class")


class CpuDevice(Device):
    name = 'cpu'
    xp = numpy

    def __repr__(self):
        return "device(type='cpu')"

    def __eq__(self, other):
        return isinstance(other, CpuDevice)

    def transfer(self, array):
        if array is None:
            return None

        if isinstance(array, numpy.ndarray):
            return array

        if numpy.isscalar(array):
            return numpy.asarray(array)
        

try:
    import cupy
    from cupy import cuda


except ImportError:
    gpu_available = False


class GpuDevice(Device):
    xp = cupy

    def __init__(self, device: cuda.Device):
        check_cuda_available()

        assert isinstance(device, cuda.Device)
        super(GpuDevice, self).__init__()
        self.device = device

    @property
    def name(self):
        return f'cupy:{self.device.id}'

    @staticmethod
    def from_device_id(device_id: int = 0):
        check_cuda_available()

        return GpuDevice(cuda.Device(device_id))

    @staticmethod
    def from_array(array: cuda.ndarray):
        if isinstance(array, cuda.ndarray) and array.deivce is not None:
            return GpuDevice(array.device)
        return None

    def __eq__(self, other):
        return isinstance(other, GpuDevice) and other.device == self.device

    def __repr__(self):
        return f"device(type='cuda', index={self.device.id})"


def is_available():
    return gpu_available


def check_cuda_available():
    if not gpu_available:
        raise RuntimeError('Install cupy first.')


def get_device(device_desc):
    '''
    根据device_desc获取设备(_Deivce)
    Args:
        device_desc: GpuDevice或CpuDevice
                     cpu -> CPU
                     cuda -> 默认显卡
                     cuda:1 -> 指定显卡1

    '''
    if isinstance(device_desc, Device):
        return device_desc

    if is_available() and isinstance(device_desc, cuda.Device):
        return GpuDevice(device_desc)

    if device_desc == 'cpu':
        return CpuDevice()

    if device_desc.startwith('cuda'):
        name, colon, device_id = device_desc.partition(':')
        if not colon:
            device_id = 0
        return GpuDevice.from_device_id(device_id)

    raise ValueError('Invalid argument.')


def get_device_from_array(array) -> Device:
    device = GpuDevice.from_array(array)
    if device is not None:
        return device

    return CpuDevice()
