# Jordan Boyd_Graber
# 2023
#
# File to define default command line parameters and to instantiate objects
# based on those parameters.  Included in most files in this project.

import logging
import argparse
import json
import gzip

from pandas import read_csv


def add_general_params(parser):
    parser.add_argument('--no_cuda', action='store_true')
    parser.set_defaults(feature=True)
    parser.add_argument('--logging_level', type=int, default=logging.INFO)
    parser.add_argument('--logging_file', type=str, default='qanta.log')
    print("Setting up logging")

def add_question_params(parser):
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--question_source', type=str, default='json')
    parser.add_argument('--questions', default = "../data/qanta.guesstrain.json",type=str)
    parser.add_argument('--secondary_questions', default = "../data/qanta.guessdev.json",type=str)
    parser.add_argument('--expo_output_root', default="expo/expo", type=str)

def add_buzzer_params(parser):
    parser.add_argument('--buzzer_guessers', nargs='+', default = ['TfidfGuesser'], help='Guessers to feed into Buzzer', type=str)
    parser.add_argument('--features', nargs='+', help='Features to feed into Buzzer', type=str,  default=['Length', 'Wiki'])    
    parser.add_argument('--buzzer_type', type=str, default="LogisticBuzzer")
    parser.add_argument('--run_length', type=int, default=100)
    parser.add_argument('--LogisticBuzzer_filename', type=str, default="models/LogisticBuzzer")    
    
def add_guesser_params(parser):
    parser.add_argument('--guesser_type', type=str, default="TfidfGuesser")
    # TODO (jbg): This is more general than tfidf, make more general (currently being used by DAN guesser as well)
    parser.add_argument('--tfidf_min_length', type=int, help="How long (in characters) must text be before it is indexed?", default=50)
    parser.add_argument('--tfidf_max_length', type=int, help="How long (in characters) must text be to be removed?", default=500)    
    parser.add_argument('--tfidf_split_sentence', type=bool, default=True, help="Index sentences rather than paragraphs")
    parser.add_argument('--wiki_min_frequency', type=int, help="How often must wiki page be an answer before it is used", default=10)
    parser.add_argument('--guesser_answer_field', type=str, default="page", help="Where is the cannonical answer")
    parser.add_argument('--TfidfGuesser_filename', type=str, default="models/TfidfGuesser")
    parser.add_argument('--WikiGuesser_filename', type=str, default="models/WikiGuesser")    
    parser.add_argument('--GprGuesser_filename', type=str, default="models/GprGuesser")
    parser.add_argument('--wiki_zim_filename', type=str, default="data/wikipedia.zim")
    parser.add_argument('--num_guesses', type=int, default=25)

    parser.add_argument('--DanGuesser_filename', type=str, default="models/DanGuesser.pkl")
    parser.add_argument('--DanGuesser_min_df', type=float, default=30,
                            help="How many documents terms must be in before inclusion in DAN vocab (either percentage or absolute count)")
    parser.add_argument('--DanGuesser_max_df', type=float, default=.4,
                            help="Maximum documents terms can be in before inclusion in DAN vocab (either percentage or absolute count)")
    parser.add_argument('--DanGuesser_min_answer_freq', type=int, default=30,
                            help="How many times we need to see an answer before including it in DAN output")
    parser.add_argument('--DanGuesser_embedding_dim', type=int, default=100)
    parser.add_argument('--DanGuesser_hidden_units', type=int, default=100)
    parser.add_argument('--DanGuesser_dropout', type=int, default=0.5)
    parser.add_argument('--DanGuesser_unk_drop', type=float, default=0.95)    
    parser.add_argument('--DanGuesser_grad_clipping', type=float, default=5.0)
    parser.add_argument('--DanGuesser_batch_size', type=int, default=128)    
    parser.add_argument('--DanGuesser_num_epochs', type=int, default=20)
    parser.add_argument('--DanGuesser_num_workers', type=int, default=0)
    

def setup_logging(flags):
    logging.basicConfig(level=flags.logging_level, force=True)
    
def load_questions(flags, secondary=False):
    question_filename = flags.questions
    if secondary:
        question_filename = flags.secondary_questions
    
    questions = None
    if flags.question_source == 'gzjson':
        logging.info("Loading questions from %s" % question_filename)
        with gzip.open(question_filename) as infile:
            questions = json.load(infile)
    
    if flags.question_source == 'json':
        with open(question_filename) as infile:
            questions = json.load(infile)
            
    if flags.question_source == 'csv':
        questions = read_csv(question_filename)

    if flags.question_source == 'expo':
        questions = ExpoQuestions()
        if flags.questions:
            questions.load_questions(question_filename)
        else:
            questions.debug()
        
    assert questions is not None, "Did not load %s of type %s" % (flags.questions, flags.question_source)

    if flags.limit > 0:
        questions = questions[:flags.limit]

    logging.info("Read %i questions" % len(questions))
        
    return questions

def instantiate_guesser(guesser_type, flags, load):
    import torch
    
    # TODO: Move this to params so that it would apply 
    cuda = not flags.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    logging.info("Using device '%s' (cuda flag=%s)" % (device, str(flags.no_cuda)))
    
    guesser = None
    logging.info("Initializing guesser of type %s" % guesser_type)
    if guesser_type == "GprGuesser":
        from gpr_guesser import GprGuesser
        logging.info("Loading %s guesser" % guesser_type)
        guesser = GprGuesser(flags.GprGuesser_filename)
        if load:
            guesser.load()
    if guesser_type == "TfidfGuesser":
        from tfidf_guesser import TfidfGuesser        
        guesser = TfidfGuesser(flags.TfidfGuesser_filename)  
        if load:                                             
            guesser.load()
    if guesser_type == "DanGuesser":                                
        from dan_guesser import DanGuesser                          
        guesser = DanGuesser(filename=flags.DanGuesser_filename, answer_field=flags.guesser_answer_field, min_token_df=flags.DanGuesser_min_df, max_token_df=flags.DanGuesser_max_df,
                    min_answer_freq=flags.DanGuesser_min_answer_freq, embedding_dimension=flags.DanGuesser_embedding_dim,
                    hidden_units=flags.DanGuesser_hidden_units, nn_dropout=flags.DanGuesser_dropout,
                    grad_clipping=flags.DanGuesser_grad_clipping, unk_drop=flags.DanGuesser_unk_drop,
                    batch_size=flags.DanGuesser_batch_size,
                    num_epochs=flags.DanGuesser_num_epochs, num_workers=flags.DanGuesser_num_workers,
                    device=device)
        if load:                                                    
            guesser.load()                                          
    if guesser_type == "PresidentGuesser":
        from president_guesser import PresidentGuesser        
        from president_guesser import training_data
        guesser = PresidentGuesser()
        guesser.train(training_data)

            
    assert guesser is not None, "Guesser (type=%s) not initialized" % guesser_type

    return guesser

def load_guesser(flags, load=False):
    """
    Given command line flags, load a guesser.  Essentially a wrapper for instantiate_guesser because we don't know the type.
    """
    return instantiate_guesser(flags.guesser_type, flags, load)

def load_buzzer(flags, load=False):
    """
    Create the buzzer and its features.
    """
    
    print("Loading buzzer")
    buzzer = None
    if flags.buzzer_type == "LogisticBuzzer":
        from logistic_buzzer import LogisticBuzzer
        buzzer = LogisticBuzzer(flags.LogisticBuzzer_filename, flags.run_length, flags.num_guesses)

    if load:
        buzzer.load()

    assert buzzer is not None, "Buzzer (type=%s) not initialized" % flags.buzzer_type


    for gg in flags.buzzer_guessers:
        guesser = instantiate_guesser(gg, flags, load=True)
        guesser.load()
        logging.info("Adding %s to Buzzer" % gg)
        buzzer.add_guesser(gg, guesser, gg==flags.guesser_type)

    print("Initializing features: %s" % str(flags.features))
    print("dataset: %s" % str(flags.questions))

    ######################################################################
    ######################################################################
    ######################################################################
    ######
    ######
    ######  For the feature engineering homework, here's where you need
    ######  to add your features to the buzzer.
    ######
    ######
    ######################################################################
    ######################################################################
    ######################################################################    
    
    for ff in flags.features:
        if ff == "Length":
            from features import LengthFeature
            feature = LengthFeature(ff)
            buzzer.add_feature(feature)
        if ff == "Wiki":
            from features import WikiFeature
            feature = WikiFeature(ff)
            buzzer.add_feature(feature)
        
    return buzzer
