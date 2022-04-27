# Question and Answer Generator: Using `Questgen`
___
Creating a custom dataset for Alight Question Answering tasks.

* **Note:** This is still under construction! We are currently looking into methods to assess the generated questions and answers.

**When you try to use the Questgen.ai, there in currently an issue**:
    we need to change the source code in **mcq.py** file from `from similarity.normalized_levenshtein import NormalizedLevenshtein` to `from strsimpy.normalized_levenshtein import NormalizedLevenshtein`.**
___

## Examples of how to use:

Instead of using the `pdf_to_clean_text` method, you may use your own method to gather clean text. There are, however, points to keep in mind: 
* The current model has maximum input token length of 512
* **Deprecated:** Divide the passage into different paragraphs/strides
* The model is currently generating the QnAs on the sentence level

* **Deprecated:** Since we are using the stride method, it is recommended to use the `clean_text()` method after you get the data formatted like `text = [paragraph1,paragraph2, ...]`. Your output would be the following: `clean_text(text) = [paragraph1,paragraph2, ...]`, and for each paragraph, last sentence of the previous paragraph would be the first sentence of the current, etc.  

There are currently 3 classes within the `QGen.py` code class, **`QuestionGenerator(text)`**, **`BoolQAnswer(df)`** , and  **`ImpossibleQuestions(df)`**.
* Update: The `get_sentences()` which is a funtion of the **`QuestionGenerator(text)`** class now automatically takes the paragraph and creates a list of sentences to prepare the input for the model.

To use `BoolQAnswer(df)`, we have to use the output generated from `QuestionGenerator(text)` or the `QuestionGenerator.df` , which is a dataframe of `'Questions','Answers_FAQ','Answers_AP','Contexts'`columns, as an input for the `BoolQAnswer(df)`.

* i.e. `BoolQAnswer(QuestionGenerator.df)`
* Keep in mind that the dataframe generated would output **`NaN`** for the non-boolean questions under the **Answers_BoolQ** and **Scores_BoolQ** columns.
    
To use `ImpossibleQuestions(df)`, we have to use the output generated from `BoolQAnswer(df)` which is the same as `BoolQAnswer.boolq_df` , which is a dataframe of `'Questions','Answers_FAQ','Answers_AP','Contexts', 'Answers_BoolQ' ,'Scores_BoolQ'`columns, as an input for the `ImpossibleQuestions(df)`.

* i.e. `ImpossibleQuestions(BoolQAnswer.boolq_df)`
* Outputs atrribute `impos_df`



```python
# Imports:


from alight_transformers.QGen import QuestionGenerator, BoolQAnswer, ImpossibleQuestions
from alight_transformers.QGen import pdf_to_clean_text
```

### Examples of how to use `pdf_to_clean_text()`:



```python
path = "path_to_file"

# We need pdf_to_clean_text(path_to_file, start_page, last_page)
text = pdf_to_clean_text(path, 10,11)
```
### Examples of how to Generate the Dataset:


```python
# For purpuses of the example, I will use a short hypothetical text:
text = ['HR teams can use Questgen to create assessments from compliance documents. Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.',
       'Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process. They can have a small in-house team and save hugely on time and cost.',
       'Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds. They can avoid repetitive questions chosen from a fixed question bank every year.']

QnA = QuestionGenerator(text)
```

    ZERO
    

    C:\Users\a1052739\Anaconda3\envs\huggingface\lib\site-packages\transformers\tokenization_utils_base.py:2198: FutureWarning: The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert).
      warnings.warn(
    C:\Users\a1052739\Anaconda3\envs\huggingface\lib\site-packages\transformers\models\t5\tokenization_t5.py:190: UserWarning: This sequence already has </s>. In future versions this behavior may lead to duplicated eos tokens being added.
      warnings.warn(
    

    Running model for generation
    {'questions': [{'Question': 'What can be generated to make sure that employees have read and understood the new policies?', 'Answer': 'assessments', 'id': 1, 'context': 'Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.'}, {'Question': 'Who can be given an assessment every time a change in policies is made?', 'Answer': 'employees', 'id': 2, 'context': 'Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.'}]}
    Running model for generation
    {'questions': [{'Question': 'What are some examples of companies that can use Questgen instead of outsourcing the assessment creation process?', 'Answer': 'textbook publishers', 'id': 1, 'context': 'Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process.'}]}
    Running model for generation
    {'questions': [{'Question': 'What is the best way to save time and money?', 'Answer': 'house team', 'id': 1, 'context': 'They can have a small in-house team and save hugely on time and cost.'}, {'Question': 'How can I save time and money by having a small in-house team?', 'Answer': 'cost', 'id': 2, 'context': 'They can have a small in-house team and save hugely on time and cost.'}]}
    Running model for generation
    {'questions': [{'Question': 'What can teachers create with Questgen?', 'Answer': 'worksheets', 'id': 1, 'context': 'Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.'}]}
    Running model for generation
    {'questions': [{'Question': 'What is the best way to avoid repetitive questions?', 'Answer': 'question bank', 'id': 1, 'context': 'They can avoid repetitive questions chosen from a fixed question bank every year.'}, {'Question': 'How often do people avoid repetitive questions from a fixed question bank?', 'Answer': 'year', 'id': 2, 'context': 'They can avoid repetitive questions chosen from a fixed question bank every year.'}]}
    ZERO
    


```python
QnA.df
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
      <td>What can be generated to make sure that employees have read and understood the new policies?</td>
      <td>assessments</td>
      <td>Assessments</td>
      <td>Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Who can be given an assessment every time a change in policies is made?</td>
      <td>employees</td>
      <td>Employees</td>
      <td>Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What are some examples of companies that can use Questgen instead of outsourcing the assessment creation process?</td>
      <td>textbook publishers</td>
      <td>Textbook publishers and edtech companies can use questgen instead of outsourcing the assessment creation process.</td>
      <td>Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What is the best way to save time and money?</td>
      <td>house team</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>How can I save time and money by having a small in-house team?</td>
      <td>cost</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>What can teachers create with Questgen?</td>
      <td>worksheets</td>
      <td>Teachers and schools can use the questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>What is the best way to avoid repetitive questions?</td>
      <td>question bank</td>
      <td>They can avoid repetitive questions chosen from a fixed question bank every year.</td>
      <td>They can avoid repetitive questions chosen from a fixed question bank every year.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>How often do people avoid repetitive questions from a fixed question bank?</td>
      <td>year</td>
      <td>Every year</td>
      <td>They can avoid repetitive questions chosen from a fixed question bank every year.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Is there a difference between true and false?</td>
      <td>None</td>
      <td>there is a difference between true and false.</td>
      <td>Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Is there such a thing as an assessment?</td>
      <td>None</td>
      <td>assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
      <td>Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Is there such a thing as an assessment of employee policies?</td>
      <td>None</td>
      <td>assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
      <td>Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Is questgen the same as testgen?</td>
      <td>None</td>
      <td>questgen is the same as testgen.</td>
      <td>Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process.</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Is questgen the same as questgen?</td>
      <td>None</td>
      <td>questgen is the same as questgen.</td>
      <td>Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process.</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Can you use questgen to create an assessment?</td>
      <td>None</td>
      <td>you can use questgen to create an assessment.</td>
      <td>Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process.</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Is it possible to have a small team in house?</td>
      <td>None</td>
      <td>they can have a small in-house team.</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Is it possible to have a small in house team?</td>
      <td>None</td>
      <td>they can have a small in-house team.</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Is it possible to have a team in house?</td>
      <td>None</td>
      <td>they can have a small in-house team.</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Can you use the questgen authoring tool in your school?</td>
      <td>None</td>
      <td>teachers and schools can use the questgen authoring tool in your school.</td>
      <td>Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Can you use questgen in a school?</td>
      <td>None</td>
      <td>teachers and schools can use the questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Is questgen true or false?</td>
      <td>None</td>
      <td>True</td>
      <td>Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Can you avoid repeating the same question every year?</td>
      <td>None</td>
      <td>they can avoid repeating the same question every year.</td>
      <td>They can avoid repetitive questions chosen from a fixed question bank every year.</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Is there such thing as a fixed question bank?</td>
      <td>None</td>
      <td>there is such thing as a fixed question bank.</td>
      <td>They can avoid repetitive questions chosen from a fixed question bank every year.</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Do you have to answer the same question every year?</td>
      <td>None</td>
      <td>they can avoid repetitive questions chosen from a fixed question bank every year.</td>
      <td>They can avoid repetitive questions chosen from a fixed question bank every year.</td>
    </tr>
  </tbody>
</table>
</div>




```python
QnA_boolq = BoolQAnswer(QnA.df)
```

    C:\Users\a1052739\Projects\Question Generator\Questgen Folder\Package\alight_transformers\QGen.py:228: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      boolq_df['Answers_BoolQ'] = bool_answers
    C:\Users\a1052739\Projects\Question Generator\Questgen Folder\Package\alight_transformers\QGen.py:229: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      boolq_df['Scores_BoolQ'] = scores
    


```python
QnA_boolq.boolq_df
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
      <th>Answers_BoolQ</th>
      <th>Scores_BoolQ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What can be generated to make sure that employees have read and understood the new policies?</td>
      <td>assessments</td>
      <td>Assessments</td>
      <td>Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Who can be given an assessment every time a change in policies is made?</td>
      <td>employees</td>
      <td>Employees</td>
      <td>Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What are some examples of companies that can use Questgen instead of outsourcing the assessment creation process?</td>
      <td>textbook publishers</td>
      <td>Textbook publishers and edtech companies can use questgen instead of outsourcing the assessment creation process.</td>
      <td>Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process.</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What is the best way to save time and money?</td>
      <td>house team</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>How can I save time and money by having a small in-house team?</td>
      <td>cost</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>What can teachers create with Questgen?</td>
      <td>worksheets</td>
      <td>Teachers and schools can use the questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>What is the best way to avoid repetitive questions?</td>
      <td>question bank</td>
      <td>They can avoid repetitive questions chosen from a fixed question bank every year.</td>
      <td>They can avoid repetitive questions chosen from a fixed question bank every year.</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>How often do people avoid repetitive questions from a fixed question bank?</td>
      <td>year</td>
      <td>Every year</td>
      <td>They can avoid repetitive questions chosen from a fixed question bank every year.</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Is there a difference between true and false?</td>
      <td>None</td>
      <td>there is a difference between true and false.</td>
      <td>Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
      <td>Yes</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Is there such a thing as an assessment?</td>
      <td>None</td>
      <td>assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
      <td>Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
      <td>Yes</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Is there such a thing as an assessment of employee policies?</td>
      <td>None</td>
      <td>assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
      <td>Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
      <td>Yes</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Is questgen the same as testgen?</td>
      <td>None</td>
      <td>questgen is the same as testgen.</td>
      <td>Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process.</td>
      <td>No</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Is questgen the same as questgen?</td>
      <td>None</td>
      <td>questgen is the same as questgen.</td>
      <td>Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process.</td>
      <td>Yes</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Can you use questgen to create an assessment?</td>
      <td>None</td>
      <td>you can use questgen to create an assessment.</td>
      <td>Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process.</td>
      <td>Yes</td>
      <td>0.98</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Is it possible to have a small team in house?</td>
      <td>None</td>
      <td>they can have a small in-house team.</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
      <td>Yes</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Is it possible to have a small in house team?</td>
      <td>None</td>
      <td>they can have a small in-house team.</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
      <td>Yes</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Is it possible to have a team in house?</td>
      <td>None</td>
      <td>they can have a small in-house team.</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
      <td>Yes</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Can you use the questgen authoring tool in your school?</td>
      <td>None</td>
      <td>teachers and schools can use the questgen authoring tool in your school.</td>
      <td>Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>Yes</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Can you use questgen in a school?</td>
      <td>None</td>
      <td>teachers and schools can use the questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>Yes</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Is questgen true or false?</td>
      <td>None</td>
      <td>True</td>
      <td>Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>Yes</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Can you avoid repeating the same question every year?</td>
      <td>None</td>
      <td>they can avoid repeating the same question every year.</td>
      <td>They can avoid repetitive questions chosen from a fixed question bank every year.</td>
      <td>Yes</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Is there such thing as a fixed question bank?</td>
      <td>None</td>
      <td>there is such thing as a fixed question bank.</td>
      <td>They can avoid repetitive questions chosen from a fixed question bank every year.</td>
      <td>Yes</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Do you have to answer the same question every year?</td>
      <td>None</td>
      <td>they can avoid repetitive questions chosen from a fixed question bank every year.</td>
      <td>They can avoid repetitive questions chosen from a fixed question bank every year.</td>
      <td>No</td>
      <td>0.99</td>
    </tr>
  </tbody>
</table>
</div>




```python
QnA_pos = ImpossibleQuestions(QnA_boolq.boolq_df)
QnA_pos.impos_df
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
      <th>Answers_BoolQ</th>
      <th>Scores_BoolQ</th>
      <th>Possible</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What can be generated to make sure that employees have read and understood the new policies?</td>
      <td>assessments</td>
      <td>Assessments</td>
      <td>Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Possible</td>
      <td>0.991926</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Who can be given an assessment every time a change in policies is made?</td>
      <td>employees</td>
      <td>Employees</td>
      <td>Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Possible</td>
      <td>0.982875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What are some examples of companies that can use Questgen instead of outsourcing the assessment creation process?</td>
      <td>textbook publishers</td>
      <td>Textbook publishers and edtech companies can use questgen instead of outsourcing the assessment creation process.</td>
      <td>Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Possible</td>
      <td>0.986688</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What is the best way to save time and money?</td>
      <td>house team</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Possible</td>
      <td>0.966886</td>
    </tr>
    <tr>
      <th>4</th>
      <td>How can I save time and money by having a small in-house team?</td>
      <td>cost</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Impossible</td>
      <td>0.935677</td>
    </tr>
    <tr>
      <th>5</th>
      <td>What can teachers create with Questgen?</td>
      <td>worksheets</td>
      <td>Teachers and schools can use the questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Possible</td>
      <td>0.981222</td>
    </tr>
    <tr>
      <th>6</th>
      <td>What is the best way to avoid repetitive questions?</td>
      <td>question bank</td>
      <td>They can avoid repetitive questions chosen from a fixed question bank every year.</td>
      <td>They can avoid repetitive questions chosen from a fixed question bank every year.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Possible</td>
      <td>0.976209</td>
    </tr>
    <tr>
      <th>7</th>
      <td>How often do people avoid repetitive questions from a fixed question bank?</td>
      <td>year</td>
      <td>Every year</td>
      <td>They can avoid repetitive questions chosen from a fixed question bank every year.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Possible</td>
      <td>0.900476</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Is there a difference between true and false?</td>
      <td>None</td>
      <td>there is a difference between true and false.</td>
      <td>Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
      <td>Yes</td>
      <td>0.89</td>
      <td>Possible</td>
      <td>0.961348</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Is there such a thing as an assessment?</td>
      <td>None</td>
      <td>assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
      <td>Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
      <td>Yes</td>
      <td>0.92</td>
      <td>Possible</td>
      <td>0.993999</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Is there such a thing as an assessment of employee policies?</td>
      <td>None</td>
      <td>assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
      <td>Every time there is a change in policies, assessments could be generated and given to employees to make sure that they have read and understood the new policies.</td>
      <td>Yes</td>
      <td>0.85</td>
      <td>Possible</td>
      <td>0.990870</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Is questgen the same as testgen?</td>
      <td>None</td>
      <td>questgen is the same as testgen.</td>
      <td>Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process.</td>
      <td>No</td>
      <td>0.72</td>
      <td>Possible</td>
      <td>0.928253</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Is questgen the same as questgen?</td>
      <td>None</td>
      <td>questgen is the same as questgen.</td>
      <td>Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process.</td>
      <td>Yes</td>
      <td>0.89</td>
      <td>Possible</td>
      <td>0.943139</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Can you use questgen to create an assessment?</td>
      <td>None</td>
      <td>you can use questgen to create an assessment.</td>
      <td>Textbook publishers and edtech companies can use Questgen instead of outsourcing the assessment creation process.</td>
      <td>Yes</td>
      <td>0.98</td>
      <td>Possible</td>
      <td>0.986241</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Is it possible to have a small team in house?</td>
      <td>None</td>
      <td>they can have a small in-house team.</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
      <td>Yes</td>
      <td>0.99</td>
      <td>Possible</td>
      <td>0.991944</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Is it possible to have a small in house team?</td>
      <td>None</td>
      <td>they can have a small in-house team.</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
      <td>Yes</td>
      <td>0.99</td>
      <td>Possible</td>
      <td>0.996165</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Is it possible to have a team in house?</td>
      <td>None</td>
      <td>they can have a small in-house team.</td>
      <td>They can have a small in-house team and save hugely on time and cost.</td>
      <td>Yes</td>
      <td>0.99</td>
      <td>Possible</td>
      <td>0.984167</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Can you use the questgen authoring tool in your school?</td>
      <td>None</td>
      <td>teachers and schools can use the questgen authoring tool in your school.</td>
      <td>Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>Yes</td>
      <td>0.99</td>
      <td>Possible</td>
      <td>0.995307</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Can you use questgen in a school?</td>
      <td>None</td>
      <td>teachers and schools can use the questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>Yes</td>
      <td>0.99</td>
      <td>Possible</td>
      <td>0.994938</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Is questgen true or false?</td>
      <td>None</td>
      <td>True</td>
      <td>Teachers and Schools can use the Questgen authoring tool to create worksheets easily in less than 5 seconds.</td>
      <td>Yes</td>
      <td>0.92</td>
      <td>Possible</td>
      <td>0.952583</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Can you avoid repeating the same question every year?</td>
      <td>None</td>
      <td>they can avoid repeating the same question every year.</td>
      <td>They can avoid repetitive questions chosen from a fixed question bank every year.</td>
      <td>Yes</td>
      <td>0.94</td>
      <td>Possible</td>
      <td>0.980597</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Is there such thing as a fixed question bank?</td>
      <td>None</td>
      <td>there is such thing as a fixed question bank.</td>
      <td>They can avoid repetitive questions chosen from a fixed question bank every year.</td>
      <td>Yes</td>
      <td>0.99</td>
      <td>Possible</td>
      <td>0.981145</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Do you have to answer the same question every year?</td>
      <td>None</td>
      <td>they can avoid repetitive questions chosen from a fixed question bank every year.</td>
      <td>They can avoid repetitive questions chosen from a fixed question bank every year.</td>
      <td>No</td>
      <td>0.99</td>
      <td>Possible</td>
      <td>0.935112</td>
    </tr>
  </tbody>
</table>
</div>


