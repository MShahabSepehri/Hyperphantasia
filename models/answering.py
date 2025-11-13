import os
import re
import time
import uuid
from datasets import load_dataset
from tqdm import tqdm
from utils import io_tools
from transformers import logging


os.environ["TOKENIZERS_PARALLELISM"] = "true"
logging.set_verbosity_error()

ROOT = io_tools.get_root(__file__, 2)
PROMPTS_DIR = f'{ROOT}/configs/models/'


class BaseAnsweringModel():
    def __init__(self, model_args_path, dataset_path, difficulty=None, dataset_name='test', device='cuda', 
                 image_first=True, max_retry=3, redo_invalids=False, local_data=False):
        self.key = None
        self.model_args_path = model_args_path
        self.dataset_path = dataset_path
        self.device = device
        self.image_first = image_first
        self.difficulty = difficulty
        self.dataset_name = dataset_name
        self.max_retry = max_retry
        self.redo_invalids = redo_invalids
        self.local_data = local_data
        self.set_init_prompt()
        self.formatting_prompt = '\n\nYou can explain your reasoning but you should specify your final answer by putting <ANSWER> before and after it like <ANSWER>FINAL_ANSWER<ANSWER> where FINAL_ANSWER is your answer. For example, if your answer is A, you should say <ANSWER>A<ANSWER>'
        self.set_model_params()
        if self.local_data:
            self.evaluate = self.local_evaluate
        else:
            self.evaluate = self.hf_evaluate

    def set_dataset(self):
        if self.local_data:
            self.dataset = io_tools.load_json(f'{self.dataset_path}/dataset.json')
            self.image_dir = io_tools.load_json(f'{self.dataset_path}/meta_data.json').get('Image_dir')
        else:
            self.dataset = load_dataset(self.dataset_path, split=self.difficulty)
            self.image_dir = 'tmp/'

    def set_model_params(self):
        args = io_tools.load_json(self.model_args_path)
        self.temperature = args.get("temperature", 0)
        self.num_beams = args.get('num_beams')
        self.max_new_tokens = args.get('max_new_tokens', 2048)
        self.top_p = args.get('top_p')
        return args

    def ask_question(self, prompt):
        pass

    def set_init_prompt(self):
        tmp = io_tools.load_json(f'{PROMPTS_DIR}/init_prompts.json')
        self.init_prompt = tmp.get(self.key, tmp.get('default'))

    def set_name(self):
        self.name = f'{self.key}_{self.dataset_name}_'

    def check_resume_dict(self, answers, name):
        tmp = False
        if name in answers.keys():
            tmp = True
            if self.redo_invalids:
                tmp = not answers.get(name).get('invalid')
        if tmp:
            return answers.get(name)
        return None
    
    def hf_evaluate(self, resume_path, save_dir, precision=2):
        self.set_dataset()
        self.set_name()
        resume = io_tools.load_resume_dict(resume_path)
        answers = {}
        save_path = self.check_folder(save_dir)
        results = None
        for sample in tqdm(self.dataset):
            name = sample.get('ID')
            tmp = self.check_resume_dict(resume, name)
            if tmp is not None:
                result_dict = tmp
                answers[name] = result_dict
                io_tools.save_json(answers, f'{save_path}/{self.name}answers.json')
            else:
                id = self.save_temp_images(sample)
                result_dict = self.sample_eval(sample)
                answers[name] = result_dict
                io_tools.save_json(answers, f'{save_path}/{self.name}answers.json')
            results = self.update_results(results, name, result_dict)
            io_tools.save_json(results, f'{save_path}/{self.name}results.json')
            self.delete_temp_images(id)
        self.delete_temp_images(source=True)
        self.print_results(results, precision=precision)
        io_tools.save_json(results, f'{save_path}/{self.name}results.json')

    def save_temp_images(self, sample):
        random_id = uuid.uuid4()
        if not os.path.exists('tmp/'):
            os.makedirs('tmp/')
        image = sample.get('Image_file')
        image.save(f'tmp/{random_id}_eval_tmp_image.jpg')
        pseudo_solution = sample.get('Pseudo_solution_file')
        pseudo_solution.save(f'tmp/{random_id}_eval_tmp_sol_image.jpg')
        sample['Image_file'] = f'{random_id}_eval_tmp_image.jpg'
        #### Test
        sample['Image_file'] = f'{random_id}_eval_tmp_sol_image.jpg'
        sample['Pseudo_solution_file'] = f'{random_id}_eval_tmp_sol_image.jpg'
        return random_id
    
    @staticmethod
    def delete_temp_images(id=None, source=False):
        if id is not None:
            if os.path.exists(f'tmp/{id}_eval_tmp_image.jpg'):
                os.remove(f'tmp/{id}_eval_tmp_image.jpg')
            if os.path.exists(f'tmp/{id}_eval_tmp_sol_image.jpg'):
                os.remove(f'tmp/{id}_eval_tmp_sol_image.jpg')
        if source:
            for file in os.listdir('tmp/'):
                os.remove(f'tmp/{file}')
            if os.path.exists('tmp/'):
                os.rmdir('tmp/')

    def local_evaluate(self, resume_path, save_dir, precision=2):
        self.set_dataset()
        self.set_name()
        resume = io_tools.load_resume_dict(resume_path)
        answers = {}
        save_path = self.check_folder(save_dir)
        results = None
        key_list = self.dataset.keys()
        for name in tqdm(key_list):
            tmp = self.check_resume_dict(resume, name)
            if tmp is not None:
                result_dict = tmp
                answers[name] = result_dict
                io_tools.save_json(answers, f'{save_path}/{self.name}answers.json')
            else:
                sample = self.dataset.get(name)
                result_dict = self.sample_eval(sample)
                answers[name] = result_dict
                io_tools.save_json(answers, f'{save_path}/{self.name}answers.json')
            results = self.update_results(results, name, result_dict)
            io_tools.save_json(results, f'{save_path}/{self.name}results.json')
        self.print_results(results, precision=precision)
        io_tools.save_json(results, f'{save_path}/{self.name}results.json')
            
    def sample_eval(self, sample):
        counter = self.max_retry
        prompt = self.generate_prompt(sample)

        while counter > 0:
            full_response = self.ask_question(prompt)
            response = self.process_answer(full_response)
            if response is not None:
                break
            counter -= 1
            print(counter)
        invalid = False
        if response is None:
            correct = False
            invalid = True
        else:
            correct = self.simple_check(response, sample.get('Answer'))
        result_dict = {
            "full_response": full_response,
            "processed_response": response,
            "answer": sample.get('Answer'),
            "category": sample.get('Category'),
            "correct": correct,
            "invalid": invalid,
            "error": not correct,
            "prompt": prompt,
        }
        return result_dict


    def format_answer(self, answer):
        if self.puzzle in ['number_grid', 'color_grid']:
            return ','.join([str(x) for x in answer])
        else:
            raise NotImplemented(self.puzzle)
        
    def process_answer(self, response):
        response = response.replace('</ANSWER>', '<ANSWER>')
        matches = re.findall(r"<ANSWER>(.*?)<ANSWER>", response)
        if len(matches) > 0:
            tmp = matches[-1]
            tmp = tmp.replace('point ', '').replace('Point ', '').replace(' ', '')
            return tmp
        if len(response) == 1:
            return response
        if ": " in response:
            return response.split(": ")[0]
        matches = re.findall(r"is (.*?)\.", response)
        if len(matches) > 0:
            return matches[-1]
        return None
    
    def generate_prompt(self, sample):
        image = ('image', f"{self.image_dir}/{sample.get('Image_file')}")
        question = sample.get('Question') + self.formatting_prompt
        question = ('text', question)
        if self.image_first:
            prompt = [image, question]
        else:
            prompt = [question, image]
        return prompt

    
    def check_folder(self, save_dir):
        if save_dir is None:
            return None
        save_path = f'{save_dir}/{self.key}'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        return save_path
    
    def simple_check(self, answer, target):
        if answer.lower() == target.lower():
            return True
        return False
    
    @staticmethod
    def print_results(results, precision=2):
        print_format = "{:<35} {:<12} {:<12} {:<12} {:<12} {:<12}"
        print(print_format.format(
            'Category',
            '# Samples',
            'Accuracy(%)',
            'Errors',
            'Corrects',
            'Invalids',            
        ))
        for category in results['category-wise'].keys():
            tmp = results.get('category-wise').get(category)
            total = int(tmp.get('correct') + tmp.get('error'))
            print(print_format.format(
                category, 
                total,
                round(tmp.get('accuracy'), precision),
                int(tmp.get('error')),
                int(tmp.get('correct')),
                int(tmp.get('invalid')),
                ))
        total = int(results.get('correct') + results.get('error') + results.get('invalid'))
        print(print_format.format(
            'All', 
            total,
            round(results.get('accuracy'), precision),
            int(results.get('error')),
            int(results.get('correct')),
            int(results.get('invalid')),
        ))
    
    @staticmethod
    def update_results(results, name, sample_result):
        correct = sample_result.get('correct')
        invalid = sample_result.get('invalid')
        category = sample_result.get('category')
        if results is None:
            results = {
                'accuracy': 0,
                'correct': 0,
                'invalid': 0,
                'error': 0,
                'invalid_samples': [],
                'error_samples': [],
                'category-wise': {},
            }
        if category not in results['category-wise'].keys():
            results['category-wise'][category] = {
                'correct': 0,
                'invalid': 0,
                'error': 0,
                'invalid_samples': [],
                'error_samples': [],
            }
        if correct:
            results['correct'] += 1
            results['category-wise'][category]['correct'] += 1
        elif invalid:
            results['invalid'] += 1
            results['category-wise'][category]['invalid'] += 1
            results['invalid_samples'].append(name)
        else:
            results['error'] += 1
            results['category-wise'][category]['error'] += 1
            results['error_samples'].append(name)
        total = results.get('correct') + results.get('error')
        total = max(total, 1)
        results['accuracy'] = results.get('correct') / total * 100
        total = results.get('category-wise').get(category).get('correct') + results.get('category-wise').get(category).get('error')
        total = max(total, 1)
        results['category-wise'][category]['accuracy'] = results['category-wise'][category].get('correct') / total * 100
        return results

    
class GPTAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global gpt
        from models import gpt
        self.key = 'gpt'
        args = super().set_model_params()
        self.deployment_name = args.get("deployment_name")
        # self.max_retries = args.get("max_retries", 3)
        self.max_tokens = args.get("max_tokens", 4096)
        self.client = gpt.get_client()

    def ask_question(self, prompt):
        response = gpt.ask_question(
                    self.client, 
                    prompt, 
                    self.init_prompt, 
                    deployment_name=self.deployment_name, 
                    temperature=self.temperature,
                    print_error=False,
                    max_retry=1,
                    max_tokens=self.max_tokens)
        if response is None:
            response = "error x"
        return response
    

class LLaVAAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global llava
        from models import llava
        self.key = 'llava'
        args = super().set_model_params()
        self.model = args.get("model")
        self.cache_dir = args.get("cache_dir")
        self.model, self.processor = llava.load_model(self.model, self.device, self.cache_dir)

    def ask_question(self, prompt):
        response = llava.ask_question(self.model, 
                                      self.processor, 
                                      prompt, 
                                      self.temperature, 
                                      self.max_new_tokens)
        return response
    

class QwenAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global qwen
        from models import qwen
        self.key = 'qwen'
        args = super().set_model_params()
        self.model_name = args.get("model_name")
        self.cache_dir = args.get("cache_dir")
        self.model, self.processor = qwen.load_model(self.model_name, self.device, self.cache_dir)

    def ask_question(self, prompt):
        response = qwen.ask_question(self.model, 
                                      self.processor, 
                                      prompt, 
                                      self.temperature, 
                                      self.max_new_tokens)
        return response

    
class ClaudeAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global claude
        from models import claude
        self.key = 'claude'
        args = super().set_model_params()
        self.deployment_name = args.get("deployment_name")
        self.client = claude.get_client()

    def ask_question(self, prompt):
        response = claude.ask_question(self.client, prompt, self.init_prompt, self.temperature, self.deployment_name, self.max_new_tokens)
        return response
    

class GeminiAnswering(BaseAnsweringModel):
    def set_model_params(self):
        global gemini
        from models import gemini
        self.key = 'gemini'
        args = super().set_model_params()
        self.deployment_name = args.get("deployment_name")
        self.model = gemini.load_model(self.init_prompt, self.temperature, self.deployment_name)

    def ask_question(self, prompt):
        flag = True
        counter = 0
        while flag:
            try:
                response = gemini.ask_question(self.model, prompt)
                flag = False
            except Exception as e:
                counter += 1
                print(counter, e)
            time.sleep(10)
        return response


class LlamaAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global llama
        from models import llama
        self.key = 'llama'
        args = super().set_model_params()
        
        model_id = args.get('model_id')
        cache_dir = args.get('cache_dir')
        self.max_new_tokens = args.get("max_new_tokens")
        model, processor = llama.load_model(model_id, self.device, cache_dir=cache_dir)

        self.model = model
        self.processor = processor

    def ask_question(self, prompt):
        response = llama.ask_question(self.model,
                                      self.processor, 
                                      prompt,
                                      temperature=self.temperature,
                                      top_p=self.top_p, 
                                      num_beams=self.num_beams,
                                      max_new_tokens=self.max_new_tokens)

        return response

class DeepSeekAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global deepseek
        from models import deepseek
        self.key = 'deepseek'
        args = super().set_model_params()
        self.model_path = args.get("model_path")
        self.max_new_tokens = args.get("max_new_tokens")
        self.do_sample = args.get("do_sample")
        cache_dir = args.get('cache_dir')
        self.model, self.processor, self.tokenizer, self.janus = deepseek.load_model(self.model_path, cache_dir)

    def ask_question(self, prompt):
        response = deepseek.ask_question(self.model, self.processor, self.tokenizer, 
                                         prompt, self.temperature, self.max_new_tokens, 
                                         self.do_sample, self.janus)
        return response
    
    def process_answer(self, response):
        response = response.replace('Reason: ', '').replace('Reason:', '')
        response = response.replace('FINAL_ANSWER', '').replace('FINALANSWER', '').replace('FINAL', '').replace(' >', '>').replace('<>', '')
        response = response.replace('</ANSWER>', '<ANSWER>').replace('<<', '<').replace('>>', '>').replace('<ANSWER><ANSWER>', '<ANSWER>')
        matches = re.findall(r"ANSWER>(.*?):(.*)<ANSWER", response)
        matches = [x[0] for x in matches]
        if len(matches) == 0:
            matches = re.findall(r"ANSWER>(.*?)<ANSWER", response)
        if len(matches) == 0:
            matches = re.findall(r"is (.?):", response)
        if len(matches) == 0:
            matches = re.findall(r"Answer:(.?)", response)
        if len(matches) == 0:
            matches = re.findall(r"ANSWER>(.?)", response)
        if len(matches) == 0:
            matches = re.findall(r"ANSWER:(.?)", response)
        if len(matches) > 0:
            tmp = matches[-1]
            tmp = tmp.replace('point ', '').replace('Point ', '').replace(' ', '')
            return tmp
        if len(response) == 1:
            return response
        if len(response) == 2 and len(response.replace(' ', '')) == 1:
            return response.replace(' ', '')
        if ": " in response:
            return response.split(": ")[0]
        matches = re.findall(r"is (.*?)\.", response)
        matches = [x for x in matches if len(x) < 15 and ' ' not in x]
        if len(matches) > 0:
            return matches[-1]
        
        options = re.findall(r".: (.*?)\n", self.question)
        letters = ['A', 'B', 'C', 'D']
        if len(options) > 3:
            tmp = None
            for let, option in zip(letters, options):
                if option in response:
                    if tmp is None:
                        tmp = f'{let}'
                    else:
                        tmp = '<ANSWER>'
            return tmp
        tmp = ''
        for let in letters:
            if (f'point {let}' in response) or (f'Point {let}' in response):
                if tmp == '':
                    tmp = f'{let}'
                else:
                    tmp = '<ANSWER>'
        if tmp == '<ANSWER>' or tmp == '':
            return None
        return tmp 

class MolmoAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global molmo
        from models import molmo
        self.key = 'molmo'
        args = super().set_model_params()
        model_id = args.get('model_id')
        cache_dir = args.get('cache_dir')
        model, processor = molmo.load_model(model_id, self.device, cache_dir=cache_dir)
        self.model = model
        self.processor = processor

    def ask_question(self, prompt):
        response = molmo.ask_question(self.model, 
                                      prompt,
                                      self.processor,
                                      self.num_beams,
                                      self.max_new_tokens,
                                      self.top_p,
                                      self.temperature)
        response = response.replace('FINAL_ANSWER:', '').replace('FINAL_ANSWER', '')
        return response
    
ANSWERING_CLASS_DICT = {
    'gpt': GPTAnswering,
    'claude': ClaudeAnswering,
    'gemini': GeminiAnswering,
    'llava': LLaVAAnswering,
    'llama': LlamaAnswering,
    'qwen': QwenAnswering,
    'deepseek': DeepSeekAnswering,
    'molmo': MolmoAnswering,
}

DEFAULT_MODEL_CONFIGS = {
    'gpt': f'{ROOT}/configs/models/gpt/gpt-4o.json',
    'claude': f'{ROOT}/configs/models/claude/3_7_sonnet.json',
    'gemini': f'{ROOT}/configs/models/gemini/2_5_pro_preview.json',
    'llava': f'{ROOT}/configs/models/llava/OV_7B.json',
    'llama': f'{ROOT}/configs/models/llama/3_2_11B.json',
    'qwen': f'{ROOT}/configs/models/qwen/VL_2_5_7B.json',
    'deepseek': f'{ROOT}/configs/models/deepseek/VL2.json',
    'molmo': f'{ROOT}/configs/models/molmo/7B.json',
}
