import boto3
from pyathena import connect
from botocore.client import ClientError
import os

s3 = boto3.resource('s3')
try:
    client = boto3.client('sts')
    account_id = client.get_caller_identity()['Account']
except:
    try:
        iam = boto3.resource('iam',
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        )

        account_id = iam.CurrentUser().arn.split(':')[4]
    except:
        raise ValueError('Credentials not found, please either connect via AWS Security Token Service (STS) or set necessary IAM environmental variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)')
        
        
my_session = boto3.session.Session()
region = my_session.region_name
athena_query_results_bucket = 'aws-athena-query-results-'+account_id+'-'+region
processed_data_bucket = 'processed-data-results-'+account_id+'-'+region
model_input_bucket = 'model-input-files'+account_id+'-'+region
s3_client = boto3.client('s3')


def get_or_create_bucket(bucket):
    try:
        s3.meta.client.head_bucket(Bucket=bucket)
    except ClientError:
        s3.create_bucket(Bucket=bucket)
        print('Creating bucket '+bucket)

get_or_create_bucket(athena_query_results_bucket)
get_or_create_bucket(processed_data_bucket)
get_or_create_bucket(model_input_bucket)

    
def upload_file(file_name, bucket=processed_data_bucket, object_name=None):
    """Upload a file to an S3 bucket
    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        try:
            object_name = file_name.split('/')[-1]
        except:
            object_name = file_name
    # Upload the file
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True
    
connection = connect(s3_staging_dir='s3://'+athena_query_results_bucket+'/athena/temp')
cursor = connection.cursor()
gluedatabase="mimiciii"
processed_db = 'mimiciiiprocessed'

def download_file(filename, target, bucket=processed_data_bucket):
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket, filename, target)
    
    
def get_s3_keys_as_generator(bucket):
    """Generate all the keys in an S3 bucket."""
    kwargs = {'Bucket': bucket}
    while True:
        resp = s3_client.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            yield obj['Key']

        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break