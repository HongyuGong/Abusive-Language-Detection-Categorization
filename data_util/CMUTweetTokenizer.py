#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
wrapper for CMU tweet tokenizer
"""
import subprocess
import shlex
import os

RUN_TOKEN_CMD = "java -XX:ParallelGCThreads=2 -Xmx500m -jar ../ark-tweet-nlp-0.3.2/ark-tweet-nlp-0.3.2.jar --just-tokenize"


# RUN_TOKEN_CMD = "java -XX:ParallelGCThreads=2 -Xmx500m -jar /afs/andrew.cmu.edu/usr14/gramadur/private/Abusive-Language-Detection-Categorization/ark-tweet-nlp-0.3.2/ark-tweet-nlp-0.3.2.jar --just-tokenize"


def _split_results(rows):
    for line in rows:
        line = line.strip()
        if (len(line) > 0):
            if line.count('\t') == 1:
                parts = line.split('\t')
                tokens = parts[0]
                orig_tokens = parts[1]
                print("tokens in split:", tokens)
                yield tokens


def _call_tokenizer(tweets, run_token_cmd=RUN_TOKEN_CMD):
    tweets_cleaned = [tw.replace('\n', ' ') for tw in tweets]
    message = "\n".join(tweets_cleaned)
    message = message.encode('utf-8')
    args = shlex.split(run_token_cmd)
    po = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = po.communicate(message)
    # print("result:", result)
    tok_result = result[0].decode("utf-8").strip('\n\n')
    tok_result = tok_result.split('\n\n')
    tok_results = [pr.split('\n') for pr in tok_result]
    # print("example tok_result:", tok_results[-1])
    return tok_results


def runtokenizer_parse(tweets, run_token_cmd=RUN_TOKEN_CMD):
    tok_raw_results = _call_tokenizer(tweets, run_token_cmd)
    tok_result = []
    for tok_raw_result in tok_raw_results:
        # tok_result.append([x for x in _split_results(tok_raw_result)])
        tok_result.extend([x for x in _split_results(tok_raw_result)])
    return tok_result


if __name__ == "__main__":
    print("Test: pass in two messages, get a list of tokens back:")
    tweets = ['@jack that\'s a Message from http://google.com', 'and a second part-of-speech \n message']
    print(runtokenizer_parse(tweets))
