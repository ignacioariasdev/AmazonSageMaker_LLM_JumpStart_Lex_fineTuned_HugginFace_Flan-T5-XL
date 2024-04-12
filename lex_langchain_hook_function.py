"""Lambda that acts as the fulfillment hook for either a Lex bot or the QnABot on AWS Solution
"""
from dispatchers.LexV2SMLangchainDispatcher import LexV2SMLangchainDispatcher
from dispatchers import utils
import logging
import os

logger = utils.get_logger(__name__)
logger.setLevel(logging.DEBUG)


def dispatch_lexv2(request):
    """Summary
    
    Args:
        request (dict): Lambda event containing an user's input chat message and context (historical conversation)
        Uses the LexV2 sessions API to manage past inputs https://docs.aws.amazon.com/lexv2/latest/dg/using-sessions.html
    
    Returns:
        dict: Description
    """
    print("About to run LexV2SMLangchainDispatcher")
    lexv2_dispatcher = LexV2SMLangchainDispatcher(request)
    return lexv2_dispatcher.dispatch_intent()

def lambda_handler(event, context):
    print(event)
    endpoint_name = os.environ['ENDPOINT_NAME']
    print(endpoint_name)
    if 'sessionState' in event:
        if 'intent' in event['sessionState']:
            if 'name' in event['sessionState']['intent']:
                if event['sessionState']['intent']['name'] == 'FallbackIntent':
                    print("dispatch_lexv2")
                    return dispatch_lexv2(event)
    else:
        print("dispatch_qnabot")
        return True
