import pyopencl as cl
import numpy as np
import sys

kernel_str = """
__kernel void raiz(__global const float *x, __global float *y) {
    int i = get_global_id(0);
    y[i] = sqrt(x[i]);
}
"""

def main():
    if len(sys.argv) != 2:
        print("Uso: python script.py <elementos>")
        sys.exit(1)

    elementos = int(sys.argv[1])
    X = np.arange(elementos, dtype=np.float32)
    Y = np.empty_like(X)

    # Inicializacao
    plataformas = cl.get_platforms()
    dispositivos = plataformas[0].get_devices(cl.device_type.ALL)
    contexto = cl.Context(dispositivos)
    fila = cl.CommandQueue(contexto, device=dispositivos[0])
    programa = cl.Program(contexto, kernel_str).build()
    kernel = cl.Kernel(programa, "raiz")

    # Preparação da memória
    bufferX = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=X)
    bufferY = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, Y.nbytes)

    # Execução
    cl.enqueue_copy(fila, src=X, dest=bufferX)
    kernel.set_args(bufferX, bufferY)
    global_size = (elementos,)
    local_size = None
    fila.enqueue_nd_range_kernel(kernel, global_size, local_size)
    fila.finish()
    cl.enqueue_copy(fila, src=bufferY, dest=Y).wait()

    # Impressão do resultado
    for i in range(elementos):
        print('[{}]'.format(Y[i]), end=' ')
    print()

if __name__ == "__main__":
    main()
