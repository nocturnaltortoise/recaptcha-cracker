import matplotlib.pyplot as plt
import json
import skimage.io
import os
import numpy as np
import textwrap
from cycler import cycler
from matplotlib.pyplot import cm
import random
import matplotlib.patches as mpatches


def graph_query_amounts(captcha_queries, query_amounts):
    queries_and_amounts = zip(captcha_queries, query_amounts)
    queries_and_amounts = sorted(queries_and_amounts, key=lambda x:x[1], reverse=True)
    captcha_queries, query_amounts = zip(*queries_and_amounts)

    # colours = cm.Dark2(np.linspace(0,1,len(captcha_queries)))
    # legend_info = zip(query_numbers, colours)
    # random.shuffle(colours)
    # captcha_queries = [textwrap.fill(query, 10) for query in captcha_queries] 
    bars = plt.bar(left=range(len(query_amounts)), height=query_amounts)
    plt.xlabel('CAPTCHA queries.')
    plt.ylabel('Query frequencies.')
    plt.xticks([])
    # plt.xticks(range(len(captcha_queries)), captcha_queries, rotation='vertical')
    
    # colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', ]

    patches = [mpatches.Patch(color=colours[j], label=captcha_queries[j]) for j in range(len(captcha_queries))]
    plt.legend(handles=patches)

    for i, bar in enumerate(bars):
        bar.set_color(colours[i])
    
    plt.show()

def graph_correct_captchas(captcha_queries, correct_captchas):
    queries_and_correct_scores = zip(captcha_queries, correct_captchas)
    queries_and_correct_scores = sorted(queries_and_correct_scores, key=lambda x:x[1], reverse=True)
    captcha_queries, correct_captchas = zip(*queries_and_correct_scores)

    captcha_queries = [textwrap.fill(query, 10) for query in captcha_queries]
    bars = plt.bar(left=range(len(correct_captchas)), height=correct_captchas)

    patches = [mpatches.Patch(color=colours[j], label=captcha_queries[j]) for j in range(len(captcha_queries))]
    plt.legend(handles=patches)
    plt.xticks([])

    for i, bar in enumerate(bars):
        bar.set_color(colours[i])

    plt.show()

# graph_correct_captchas(captcha_queries, correct_captchas)
# graph_query_amounts(captcha_queries, query_amounts)

def show_checkbox_predictions(checkboxes, rows, cols, captcha_query, correct):

    fig, axes = plt.subplots(rows, cols)
    for checkbox in checkboxes:
        position = checkbox['position']
        path = checkbox['path']
        predictions = checkbox['predictions']
        matching = checkbox['matching']

        if path:
            path = os.path.join('../', path)
            x = position[0]-1
            y = position[1]-1
            axes[x, y].imshow(skimage.io.imread(path))
            if matching:
                axes[x, y].set_title("Picked \n {0}".format(predictions))
            else:
                axes[x, y].set_title(predictions)

            axes[x, y].set_xticks([])
            axes[x, y].set_yticks([])

    fig.suptitle("{0}, Correct".format(captcha_query)) if correct else fig.suptitle("{0}, Incorrect".format(captcha_query))
    plt.subplots_adjust(hspace=0.5)

    plt.show() 


def load_guess_file(captcha_query=None):
    with open('../0.1-probability-4.8-guesses.json', 'r') as predictions_file:
        predictions = predictions_file.read()
        json_predictions = json.loads(predictions)

    return json_predictions

def get_captcha_data(json_predictions, captcha_query, display_predictions=False):
    query_predictions = json_predictions[captcha_query]
    query_captcha_amount = len(query_predictions)
    query_amounts.append(query_captcha_amount)
    captcha_queries.append(captcha_query)

    correct_count = 0
    for subfolder, subfolder_contents in query_predictions.items():
        correct = subfolder_contents['correct']
        if correct:
            correct_count += 1

        rows = int(subfolder_contents['rows'])
        cols = int(subfolder_contents['cols'])
        checkboxes = subfolder_contents['checkboxes']

        if display_predictions:
            show_checkbox_predictions(checkboxes, rows, cols, captcha_query, correct)

    correct_captchas.append(correct_count)

plt.rc('font', size=10)
# plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y'])))
# CAPTCHA_QUERY = 'store front'
CAPTCHA_QUERY = None
json_predictions = load_guess_file()
colours = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 
            'darkolivegreen', 'orangered', 'rebeccapurple', 
            'cadetblue', 'goldenrod', 'darksalmon', 
            'darkmagenta', 'springgreen', 'plum']

query_amounts = []
captcha_queries = []
correct_captchas = []

if CAPTCHA_QUERY:
    get_captcha_data(json_predictions, CAPTCHA_QUERY, display_predictions=True)

else:
    total_captchas = 0
    for captcha_query in json_predictions:
        total_captchas += len(json_predictions[captcha_query])
        get_captcha_data(json_predictions, captcha_query)
    print(total_captchas)
    graph_query_amounts(captcha_queries, query_amounts)
    print(sum(correct_captchas))
    graph_correct_captchas(captcha_queries, correct_captchas)
    
    

    