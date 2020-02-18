aws autoscaling describe-auto-scaling-instances --region us-east-1 --output text --query "AutoScalingInstances[?AutoScalingGroupName=='ASG-GROUP-NAME'].InstanceId" | xargs -n1 aws ec2 describe-instances --instance-ids $ID --region us-east-1 --query "Reservations[].Instances[].PrivateIpAddress" --output text >temp_hosts
sed 's/\t\t*/\n/g' temp_hosts >hosts
python3 -m scoop --hostfile hosts main.py
