#!/bin/bash
function run() {
    while :
    do
        eval $1
    done
}
LOCAL_HOSTNAME=$(hostname -d)
if [[ ${LOCAL_HOSTNAME} =~ .*\.amazonaws\.com ]]
then
    echo "This is an EC2 instance"
    aws autoscaling describe-auto-scaling-instances --region us-east-1 --output text --query "AutoScalingInstances[?AutoScalingGroupName=='ASG-GROUP-NAME'].InstanceId" | xargs -n1 aws ec2 describe-instances --instance-ids $ID --region us-east-1 --query "Reservations[].Instances[].PrivateIpAddress" --output text >temp_hosts
    sed 's/\t\t*/\n/g' temp_hosts >hosts
    run "python3.7 -m scoop --hostfile hosts main.py"
else
    echo "This is not an EC2 instance, or a reverse-customized one"
    run "python3.7 -m scoop main.py"
fi
