import os


def patch_onnx_runtime(
    inter_op_num_threads: int = 16,
    intra_op_num_threads: int = 16,
    omp_num_threads: int = 16,
):
    """
    Patch ONNX Runtime's defaults to set the number of threads for inter-op,
    intra-op, and OpenMP.Trying to use an ONNX Runtime session within a fal app
    without explicitly setting these parameters can lead to issues, for example,
    it can cause several logs related to these parameters to be printed.
    Please run this function before importing any ONNX Runtime modules
    in your application.

    Args:
        inter_op_num_threads (int): Number of threads for inter-op parallelism.
        intra_op_num_threads (int): Number of threads for intra-op parallelism.
        omp_num_threads (int): Number of threads for OpenMP parallelism.

    """
    import onnxruntime as ort

    os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    _default_session_options = ort.capi._pybind_state.get_default_session_options()

    def get_default_session_options_new():
        _default_session_options.inter_op_num_threads = inter_op_num_threads
        _default_session_options.intra_op_num_threads = intra_op_num_threads
        return _default_session_options

    ort.capi._pybind_state.get_default_session_options = get_default_session_options_new
