
summary.txt: benchmark_blueqat.txt benchmark_cirq.txt benchmark_qiskit.txt benchmark_sympy.txt
	(echo '==== Blueqat ====' && \
	 egrep 'TIME_ADD|TIME_SUB|TIME_MODULOADD' benchmark_blueqat.txt && \
	echo '==== Cirq ====' && \
	 egrep 'TIME_ADD|TIME_SUB|TIME_MODULOADD' benchmark_cirq.txt && \
	echo '==== Qiskit ====' && \
	 egrep 'TIME_ADD|TIME_SUB|TIME_MODULOADD' benchmark_qiskit.txt && \
	echo '==== SymPy ====' && \
	 egrep 'TIME_ADD|TIME_SUB|TIME_MODULOADD' benchmark_sympy.txt \
	) > summary.txt

benchmark_blueqat.txt: benchmark_blueqat.py
	python benchmark_blueqat.py | tee benchmark_blueqat.txt
benchmark_cirq.txt: benchmark_cirq.py
	python benchmark_cirq.py | tee benchmark_cirq.txt
benchmark_qiskit.txt: benchmark_qiskit.py
	python benchmark_qiskit.py | tee benchmark_qiskit.txt
benchmark_sympy.txt: benchmark_sympy.py
	python benchmark_sympy.py | tee benchmark_sympy.txt

