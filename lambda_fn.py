import json
import boto3


def lambda_handler(event, context):
    runtime = boto3.Session().client('sagemaker-runtime')
    response = runtime.invoke_endpoint(EndpointName='pytorch-inference-2022-02-23-05-44-04-881',
                                       ContentType='text/csv',
                                       Accept='text/csv',
                                       Body=event['body'])
    result = response['Body'].read().decode('utf-8')

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/csv',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'DELETE, POST, GET, OPTIONS',
                    'Access-Control-Allow-Headers': '*'},
        'body': result
    }
