import pyopencl as cl
import numpy as np

kernel_str = """
__kernel void raiz(__global const float *x, __global float *y) {
    int i = get_global_id(0);
    y[i] = sqrt(x[i]);
}
"""

def main():
    elementos = 10  # Definindo um número fixo de elementos para evitar a entrada do usuário
    X = np.array([float(i) for i in range(elementos)], dtype=np.float32)
    Y = np.empty_like(X)

    # Inicialização
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices(cl.device_type.ALL)
    context = cl.Context(devices)
    queue = cl.CommandQueue(context, device=devices[0])

    program = cl.Program(context, kernel_str).build()

    # Preparação da memória
    bufferX = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=X)
    bufferY = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, Y.nbytes)

    # Execução do kernel
    program.raiz(queue, (elementos,), None, bufferX, bufferY)
    cl.enqueue_copy(queue, Y, bufferY).wait()

    # Exibição dos resultados
    print("Resultados:")
    for result in Y:
        print(result)

if __name__ == "__main__":
    main()
