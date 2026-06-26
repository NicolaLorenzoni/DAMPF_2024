#!/bin/bash
##### RESOURCES ALLOCATION (SLURM) #############
### THIS ARE THE EDITABLE PARAMETERS ###########
# In here we give the job a name
#SBATCH --job-name=FMO_77K_MARCH
#
# In here we specify the required nodes
#SBATCH --nodes=1
#
# In here we specify the tasks per node (HAS TO BE 1)
# We want a single ray instance per node
#SBATCH --ntasks-per-node=1
#
# In here we specify the number of cores for each task
# Yep, cores. So on Justus 2 up to 96 per node.
#SBATCH --cpus-per-task=48
#
# In here we specify the memory per node (GiB units)
#SBATCH --mem=250GB
#
# Send mail when job begins, aborts and ends
#SBATCH --mail-type=ALL
#
# In here we specify the required time
#SBATCH --time=200:00:00
###############################################

#### RAY MEMORY SETTINGS #######################
# Ray Cpus are the cores. On Justus2 we have 2 cores per cpu
RAY_CPUS_PER_NODE=$((${SLURM_CPUS_PER_TASK}*2))
# We set the memory available to Ray equal to the one we specified 
# in the resources allocation. Note that SLURM_MEM_PER_NODE gives a 
# result in Mb while ray wants it in bytes, so we rescale.
# The following are int operations. Do not write float numbers.
RAY_MEM=$((${SLURM_MEM_PER_NODE}*1024*1024)) 
RAY_OBJ_MEM=$((${RAY_MEM}*2/3))
############################################

#### LOADING ENVIRONMENT CONTAINING RAY #################
# Load modules or your own conda environment here
source ~/.bashrc
conda activate ray
#########################################################


################# GENERATION OF REDIS PASSWORD AND NODES NAMES ##################
# Redis unique passwrod generation
redis_password=$(uuidgen)
export redis_password
# Storing allocated nodes names
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) 
nodes_array=( $nodes )
# Storing main node's name
headnode=${nodes_array[0]} 
# Running task on it to obtain ip
ip=$(srun --nodes=1 --ntasks=1 -w $headnode hostname --ip-address) 
# Define a port on which connect to the head node
port=6379
# Ip of head node for secondary nodes
ip_head=$ip:$port
export ip_head
#################################################################################



##### STARTING RAY INSTANCES ON EVERY NODE ################################
echo "STARTING HEAD at $headnode"
# We start with the head node and block it to maintain the ray instance operative
srun --nodes=1 --ntasks=1 -w $headnode \
    ray start  --num-cpus=${RAY_CPUS_PER_NODE} --memory=${RAY_MEM} --object-store-memory=${RAY_OBJ_MEM} --head --port=$port --redis-password=$redis_password --block &
# We wait some time to be sure the instance has been started
sleep 30
# Now we boot also the other nodes
side_nodes=$(($SLURM_JOB_NUM_NODES - 1))
for ((  i=1; i<=$side_nodes; i++ ))
do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w $node_i \
      ray start --num-cpus=${RAY_CPUS_PER_NODE} --memory=${RAY_MEM} --object-store-memory=${RAY_OBJ_MEM} --address=$ip_head --redis-password=$redis_password --block &
  sleep 5
done
echo "Ray instances initialized"
##########################################################################

#### DISPLAYING INFO (SAVED IN SLURM-JOBID.OUT) #####################
echo "Submit Directory:                     $SLURM_SUBMIT_DIR"
echo "Working Directory:                    $PWD"
echo "Running on host                       $HOSTNAME"
echo "IP Head: 			     $ip_head"
echo "Job id:                               $SLURM_JOB_ID"
echo "Job name:                             $SLURM_JOB_NAME"
echo "Number of nodes allocated to job:     $SLURM_JOB_NUM_NODES"
echo "Number of cores allocated to job:     $SLURM_NPROCS"
echo "ray memory:                           $RAY_MEM"
echo "ray object store memory:              $RAY_OBJ_MEM"
#####################################################################

#### LAUNCHING CODE ####################
# We pass to the program the number of cores
python dampfdt3.py ${RAY_CPUS_PER_NODE} ${SLURM_JOB_NUM_NODES} ${RAY_MEM} > ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log 2>&1
exit
#########################################
