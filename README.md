
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

For SymPy, I emulated 'measure' operation by using random-choice of 'measure_all' results.

Summary
-------

| Simulator  | TIME_ADD    | TIME_SUB    | TIME_MODULOADD |
| ---------- | ----------: | ----------: | -------------: |
| Blueqat    |   1.846 sec |   1.719 sec |  34.623 sec    |
| Cirq       |  49.281 sec |  49.234 sec | 388.213 sec    |
| Qiskit     |   0.204 sec |   0.130 sec |   0.943 sec    |
| SymPy      | 199.545 sec | 199.848 sec | 430.502 sec    |
