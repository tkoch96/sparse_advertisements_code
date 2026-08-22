#!/usr/bin/env bash
# ============================================================================
# Tear down the SCULPTOR Ray cluster and verify nothing is left billing.
#
# Usage:
#   ./cluster/teardown.sh                 # uses cluster/ray-cluster.yaml in current dir
#   ./cluster/teardown.sh path/to.yaml    # explicit yaml
#
# Exits non-zero if any sculptor-tagged resources are left running. Safe to
# run multiple times.
# ============================================================================

set -euo pipefail

CLUSTER_YAML="${1:-cluster/ray-cluster.yaml}"
REGION="${AWS_REGION:-us-east-1}"

if [ ! -f "$CLUSTER_YAML" ]; then
    echo "ERROR: cluster yaml not found at $CLUSTER_YAML" >&2
    exit 2
fi

echo "==> ray down  (terminates the head; workers shut down via autoscaler)"
ray down -y "$CLUSTER_YAML" || \
    echo "ray down exited non-zero; continuing to orphan check anyway..."

echo ""
echo "==> Checking for sculptor-tagged EC2 instances still running..."
ORPHANS=$(aws ec2 describe-instances \
    --region "$REGION" \
    --filters \
        "Name=tag:project,Values=sculptor" \
        "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceType]' \
    --output text)

if [ -n "$ORPHANS" ]; then
    echo ""
    echo "!!! ORPHAN INSTANCES — THESE ARE BILLING YOU !!!"
    echo "$ORPHANS"
    echo ""
    echo "Kill them with:"
    echo "  aws ec2 terminate-instances --region $REGION --instance-ids <ID> [<ID>...]"
    STATUS=1
else
    echo "OK: no sculptor-tagged instances running."
    STATUS=0
fi

echo ""
echo "==> Checking for unattached EBS volumes tagged project=sculptor..."
ORPHAN_VOLS=$(aws ec2 describe-volumes \
    --region "$REGION" \
    --filters \
        "Name=tag:project,Values=sculptor" \
        "Name=status,Values=available" \
    --query 'Volumes[].[VolumeId,Size,State]' \
    --output text)

if [ -n "$ORPHAN_VOLS" ]; then
    echo "!!! UNATTACHED EBS VOLUMES (billed while idle):"
    echo "$ORPHAN_VOLS"
    echo "Delete with: aws ec2 delete-volume --region $REGION --volume-id <ID>"
    STATUS=1
else
    echo "OK: no orphan EBS volumes."
fi

echo ""
echo "==> Checking for Elastic IPs tagged project=sculptor..."
EIPS=$(aws ec2 describe-addresses \
    --region "$REGION" \
    --filters "Name=tag:project,Values=sculptor" \
    --query 'Addresses[].[PublicIp,AssociationId,AllocationId]' \
    --output text 2>/dev/null || true)

if [ -n "$EIPS" ]; then
    echo "!!! Elastic IPs allocated (billed when not attached):"
    echo "$EIPS"
    echo "Release with: aws ec2 release-address --region $REGION --allocation-id <ID>"
    STATUS=1
else
    echo "OK: no Elastic IPs."
fi

echo ""
if [ "$STATUS" -eq 0 ]; then
    echo "All clear. Teardown complete."
else
    echo "Teardown FOUND ORPHANS — see warnings above and clean up manually."
fi
exit $STATUS
