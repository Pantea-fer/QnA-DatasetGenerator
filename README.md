# Question and Answer Generator :using `Questgen`
___
Creating a custom dataset for Alight Question Answering tasks.

* **Note:** This is still under construction! We are currently looking into methods to assess the generated questions and answers.

**When you try to use the Questgen.ai, there in currently an issue**:
    we need to change the source code in **mcq.py** file from `from similarity.normalized_levenshtein import NormalizedLevenshtein` to `from strsimpy.normalized_levenshtein import NormalizedLevenshtein`.**
___

## Examples of how to use:

Instead of using the `pdf_to_clean_text` method, you may use your own method to gather clean text. There are, however, points to keep in mind: 
* The current model has maximum input token length of 512
* Divide the passage into different paragraphs/strides
* The less sentences each paragraph includes, the more useful questions would be generated
* The format of the clean text should be a list of paragraphs as the following:
    * `clean_text = [paragraph1,paragraph2, ...]` whereas each paragraph is : `paragraph = 'sentence1, sentence2,...'`
    


```python
# Imports:

from alight_transformers.QGen import QuestionGenerator as qg
from alight_transformers.QGen import pdf_to_clean_text
```

### Examples of how to use `pdf_to_clean_text()`:



```python
path = "C:/Alight/Alight Help/HSA FAQs.pdf"

# We need pdf_to_clean_text(path_to_file, start_page, last_page)
text = pdf_to_clean_text(path, 10,11)
```


```python
# For purpuses of the example, I will use a short hypothetical text:
text = ['HR teams can use Questgen to create assessments from compliance documents. Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.',
       'Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process. They can have a small in-house team and save hugely on time and cost.',
       'Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds. They can avoid repetitive questions chosen from a fixed question bank every year.']

QnA = qg(text)
```

    C:\Users\a1052739\Anaconda3\envs\huggingface\lib\site-packages\transformers\tokenization_utils_base.py:2198: FutureWarning: The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert).
      warnings.warn(
    C:\Users\a1052739\Anaconda3\envs\huggingface\lib\site-packages\transformers\models\t5\tokenization_t5.py:190: UserWarning: This sequence already has </s>. In future versions this behavior may lead to duplicated eos tokens being added.
      warnings.warn(
    

    Running model for generation
    {'questions': [{'Question': 'Who can be given an assessment every time a change in policies is made?', 'Answer': 'employees', 'id': 1, 'context': 'Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.'}]}
    Running model for generation
    {'questions': [{'Question': 'What are some examples of companies that can use Questgen instead of outsourcing the assessment creation process?', 'Answer': 'textbook publishers', 'id': 1, 'context': 'Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process.'}]}
    Running model for generation
    {'questions': [{'Question': 'What can teachers create with Questgen?', 'Answer': 'worksheets', 'id': 1, 'context': 'Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.'}, {'Question': 'How long does it take to create a worksheet?', 'Answer': 'seconds', 'id': 2, 'context': 'Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.'}, {'Question': 'What is Questgen?', 'Answer': 'tool', 'id': 3, 'context': 'Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.'}]}
    


```python
df = QnA.create_df()
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Questions</th>
      <th>Answers_FAQ</th>
      <th>Answers_AP</th>
      <th>Contexts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Who can be given an assessment every time a change in policies is made?</td>
      <td>employees</td>
      <td>Employees</td>
      <td>Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What are some examples of companies that can use Questgen instead of outsourcing the assessment creation process?</td>
      <td>textbook publishers</td>
      <td>Textbook publishers and edtech companies can use questgen instead of outsourcing the assessment creation process.</td>
      <td>Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What can teachers create with Questgen?</td>
      <td>worksheets</td>
      <td>Teachers and schools can use the questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>How long does it take to create a worksheet?</td>
      <td>seconds</td>
      <td>5 seconds</td>
      <td>Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>What is Questgen?</td>
      <td>tool</td>
      <td>The questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Can hr teams use questgen to create compliance assessments?</td>
      <td>None</td>
      <td>Yes, hr teams can use questgen to create compliance assessments.</td>
      <td>HR teams can use Questgen to create assessments from compliance documents. Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Can hr teams use questgen to create compliance documents?</td>
      <td>None</td>
      <td>Yes, hr teams can use questgen to create compliance documents.</td>
      <td>HR teams can use Questgen to create assessments from compliance documents. Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Can hr teams use questgen to create compliance reports?</td>
      <td>None</td>
      <td>Yes, hr teams can use questgen to create compliance reports.</td>
      <td>HR teams can use Questgen to create assessments from compliance documents. Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Is questgen the same as testgen?</td>
      <td>None</td>
      <td>Yes, questgen is the same as testgen.</td>
      <td>Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process. They can have a small in-house team and save hugely on time and cost.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Is questgen the same as questgen?</td>
      <td>None</td>
      <td>Yes, questgen is the same as questgen.</td>
      <td>Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process. They can have a small in-house team and save hugely on time and cost.</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Can you use questgen to create an assessment?</td>
      <td>None</td>
      <td>Yes, you can use questgen to create an assessment.</td>
      <td>Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process. They can have a small in-house team and save hugely on time and cost.</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Can i use questgen to create a worksheet?</td>
      <td>None</td>
      <td>Yes, teachers and schools can use the questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds. They can avoid repetitive questions chosen from a fixed question bank every year.</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Can you use questgen to create a worksheet?</td>
      <td>None</td>
      <td>Yes, teachers and schools can use the questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds. They can avoid repetitive questions chosen from a fixed question bank every year.</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Can i use questgen to create worksheets?</td>
      <td>None</td>
      <td>Yes, teachers and schools can use the questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds. They can avoid repetitive questions chosen from a fixed question bank every year.</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
