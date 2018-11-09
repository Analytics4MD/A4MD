import flow

class CaliburnCluster(flow.environment.DefaultSlurmEnvironment):
    hostname_pattern = 'cal-login*'
    template = 'caliburncluster.sh'
    cores_per_node = 18
