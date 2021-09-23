
Simple Quantum Simulator Benchmark
----------------------------------

Target simulators are:
- Blueqat
- Cirq
- Qiskit
- SymPy

Benchmark circuits are:
- TIME_ADD - Execution time of quantum adder circuit (4bit + 4bit = 5bit, measure 10 times)
- TIME_SUB - Execution time of quantum subtractor circuit (4bit - 4bit = 5bit, measure 10 times)
- TIME_MUL - Execution time of quantum multiplier circuit (3bit * 3bit = 6bit, measure 10 times)
- TIME_MODULOADD - Execution time of quantum modulo-adder circuit (3bit + 3bit mod 3bit = 4bit, measure 10 times)

For SymPy, I emulated 'measure' operation by using random-choice of 'measure_all' results.

Summary
-------

| Simulator  | TIME_ADD    | TIME_SUB    | TIME_MUL    | TIME_MODULOADD |
| ---------- | ----------: | ----------: | ----------: | -------------: |
| Blueqat    |   1.731 sec |   1.720 sec |   6.313 sec |  34.661 sec    |
| Cirq       |  49.173 sec |  49.148 sec | 100.250 sec | 389.615 sec    |
| Qiskit     |   0.190 sec |   0.134 sec |   0.293 sec |   0.940 sec    |
| SymPy      | 199.883 sec | 199.850 sec | 149.376 sec | 435.899 sec    |
