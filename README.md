
Simple Quantum Simulator Benchmark
----------------------------------

Target simulators are:
- Blueqat
- Cirq
- Qiskit
- SymPy

Benchmark circuits are:
- TIME_ADD - Execution time of quantum adder circuit
- TIME_SUB - Execution time of quantum subtractor circuit
- TIME_MODULOADD - Execution time of quantum modulo-adder circuit
- TIME_MUL - Execution time of quantum multiplier circuit

For SymPy, I emulated 'measure' operation by using random-choice of 'measure_all' results.

Summary
-------

| Simulator  | TIME_ADD    | TIME_SUB    | TIME_MODULOADD | TIME_MUL    |
| ---------- | ----------: | ----------: | -------------: | ----------: |
| Blueqat    |   1.731 sec |   1.720 sec |  34.661 sec    |   6.313 sec |
| Cirq       |  49.173 sec |  49.148 sec | 389.615 sec    | 100.250 sec |
| Qiskit     |   0.190 sec |   0.134 sec |   0.940 sec    |   0.293 sec |
| SymPy      | 199.883 sec | 199.850 sec | 435.899 sec    | 149.376 sec |
