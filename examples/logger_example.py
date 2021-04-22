# Example file for transparently logging both MPI and serial jobs
import logging
from simsopt.util.logging import initialize_logging

# Serial logging
initialize_logging(filename='serial.log')
for i in range(2):
    logging.info("Hello (times %i) from serial job" % (i+1))

# MPI logging
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except:
    comm = None

if comm is not None:
    initialize_logging(mpi=True, filename='mpi.log')
    for i in range(2):
        logging.warning("Hello (times %i) from mpi job" % (i+1))
