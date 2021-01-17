
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[59]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[60]:


df = pd.read_csv('ab_data.csv')
df.head()


# b. Use the below cell to find the number of rows in the dataset.

# In[61]:


df.shape #294478 represents # of rows and 5 represents # of columns


# c. The number of unique users in the dataset.

# In[62]:


df['user_id'].nunique()


# d. The proportion of users converted.

# In[63]:


df_new = df.drop_duplicates('user_id')
df_new['converted'].sum()/df_new.shape[0]


# e. The number of times the `new_page` and `treatment` don't line up.

# In[64]:


df['landing_page'].nunique(), df['group'].nunique()


# In[65]:


df_1 = df.query('landing_page == "new_page"')
df_2 = df.query('landing_page == "old_page"')
df_1_unmatch = df_1.query('group != "treatment"')
df_2_unmatch = df_2.query('group != "control"')
df_1.query('group != "treatment"').shape[0] + df_2.query('group != "control"').shape[0]


# In[66]:


df_1_unmatch.shape[0] + df_2_unmatch.shape[0]


# f. Do any of the rows have missing values?

# In[67]:


df.isnull().sum()


# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[68]:


df2 = df.drop(df.query('(group == "treatment" and landing_page != "new_page") or (group != "treatment" and landing_page == "new_page") or (group == "control" and landing_page != "old_page") or (group != "control" and landing_page == "old_page")').index)


# In[69]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[70]:


df2['user_id'].nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[71]:


df2[df2['user_id'].duplicated()]['user_id']


# c. What is the row information for the repeat **user_id**? 

# In[72]:


df2[df2['user_id'].duplicated()]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[73]:


df2 = df2.drop(2893)


# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[74]:


df2['converted'].mean()


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[75]:


control_converted = df2[df2['group']=='control']['converted'].mean()
control_converted


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[76]:


treatment_converted = df2[df2['group']=='treatment']['converted'].mean()
treatment_converted


# d. What is the probability that an individual received the new page?

# In[77]:


len(df2.query('landing_page == "new_page"'))/len(df2)


# e. Use the results in the previous two portions of this question to suggest if you think there is evidence that one page leads to more conversions?  Write your response below.

# In[78]:


std = df2[df2['group']=='control']['converted'].std()
diff_std = (control_converted - treatment_converted) / std
std, diff_std


# **Both treatment group and control group have similar probability of 12% of converting. Therefore, there is no evidence to suggest that one page leads to more conversions.**
# 
# **Also, it seemed to be that an individual receiving new page or old page have the same probability. Therefore, it ensures distribution of old and new pages are equal. **
# 
# **The probability of conversion for treatment group is even less than probability of conversting regardless of page. In addition, with the standard deviation of 0.325, the difference is only 0.005  standard deviations.  As a result, one can conclude, there is no practical significane to investing resources in creating a new page.** 

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# $H_0:  $**$p_{new}$** - **$p_{old}$** ≤ 0
# 
# 
# $H_1:  $**$p_{new}$** - **$p_{old}$** > 0

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[79]:


p_new = df2['converted'].mean()
p_new


# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# In[80]:


p_old = df2['converted'].mean()
p_old


# c. What is $n_{new}$?

# In[81]:


n_new = len(df2.query('landing_page == "new_page"'))
n_new


# d. What is $n_{old}$?

# In[82]:


n_old = len(df2.query('landing_page == "old_page"'))
n_old


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[83]:


new_page_converted = np.random.choice([0,1],n_new,p=[1 - p_new, p_new])


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[84]:


old_page_converted = np.random.choice([0,1],n_old,p=[1-p_old,p_old])


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[85]:


p_new_old = new_page_converted.mean() - old_page_converted.mean()
p_new_old


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in **p_diffs**.

# In[86]:


p_diffs = []
for i in range(10000):
    new_page_converted = np.random.choice([0,1],n_new,p=[1 - p_new, p_new])
    old_page_converted = np.random.choice([0,1],n_old,p=[1-p_old,p_old])
    diffs = new_page_converted.mean() - old_page_converted.mean()
    p_diffs.append(diffs)

p_diffs = np.array(p_diffs)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[87]:


plt.hist(p_diffs)
plt.xlabel('probability difference between conversion rates for new and old page')
plt.ylabel('# of Occurance')
plt.title('Plot of 10000 p_diffs simulation');


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[88]:


p_actual_diff = df2.query('group == "treatment"')['converted'].mean() - df2.query('group == "control"')['converted'].mean()
prop_diff = len(p_diffs[p_diffs > p_actual_diff])/len(p_diffs)
prop_diff, p_actual_diff


# In[89]:


plt.hist(p_diffs)
plt.axvline(p_actual_diff,color='red')
plt.xlabel('probability difference between conversion rates for new and old page')
plt.ylabel('# of Occurance')
plt.title('Plot of 10000 p_diffs simulation');


# k. In words, explain what you just computed in part **j.**.  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **The proportion of p_diffs greater than the actual differernce is called p_value. The p_value calculated for this scenario is 90%.If null hypothesis is true, probability of obtaining the observed statistics is 90%. This indicates that we fail to reject the null hypothesis. This in turn concludes, there is no difference between conversion rates for the new and old pages.**

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[90]:


import statsmodels.api as sm

convert_old = df2.query('group == "control"')['converted'].sum()
convert_new = df2.query('group == "treatment"')['converted'].sum()
n_old = len(df2.query('landing_page == "new_page"'))
n_new = len(df2.query('landing_page == "old_page"'))
convert_old,convert_new,n_old,n_new


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[91]:


z_score,p_value = sm.stats.proportions_ztest([convert_old,convert_new],[n_old,n_new],alternative = 'smaller')
z_score,p_value


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# In[92]:


from scipy.stats import norm

print(norm.cdf(z_score))
# Tells us how significant our z-score is
print(norm.ppf(1-(0.05/2)))
# Tells us what our critical value at 95% confidence is


# **Answer to part n:** 
# 
# **The test data has z-score of 1.26, which is less than the critical value of 1.96. Therefore, we fail to reject the null hypothesis.**
# 
# **Also, p_value, 0.90 found using scipy library and manually method are same.**
# 
# **The conclusion found in part n is in agreement with the findings in parts j and k. We can safely conclude that there is no statistical difference between the conversion rates for the control and treatment groups.**
# 
# **The actual difference in treatment between conversion rates for the new and old page is -0.00157, while the sample difference is -0.00055. Both values from sample and actual difference are negative and similar (0.001). Therefore, one can also conclude the old and new pages have no practical significance. The difference of 0.001 yields no practical significance.**

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Logistic Regression. This is because, the response variable is categorical in nature. The response variable here is if there is difference between the conversion rates for the new page and old page. In Linear regression, the dependent variable is continuous and could have infinite number of possible outcomes.**

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a colun for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[93]:


df2['intercept'] = 1
df2[['control','treatment']] = pd.get_dummies(df['group'])


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[94]:


log_mod = sm.Logit(df2['converted'],df2[['intercept','treatment']])
results = log_mod.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[95]:


results.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in the **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the **Part II**?

# **P value associated with logistic regression model is 0.190. P value is greater than alpha value of 0.05. That means, treatment is not significant in making the user convert. The previous numerical value of p (0.90) is differnt from logistic regression p value (0.190). This is because there is difference in the null and alternative hypotheses associated with regression model and Part II.**
# 
# **For example,** 
# 
# **Part II(A/B Testing):** 
# 
# $H_0:  $**$p_{new}$** - **$p_{old}$** ≤ 0
# 
# $H_1:  $**$p_{new}$** - **$p_{old}$** > 0
# 
# **Logistic Regression Model:**
# 
# $H_0:  $**$p_{new}$** - **$p_{old}$** = 0
# 
# $H_1:  $**$p_{new}$** - **$p_{old}$** != 0

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# In[96]:


df2.head()
type(df2['timestamp'][0])
df2.head()
df2['timestamp'].max(),df2['timestamp'].min()


# **We can consider timestamp, that might influence whether or not an individual converts. It is a good idea to consider other factors into our regression model. This is because, we can get develop more accurate model, considering all different criteria to predict specific outcome. For example, time can play a role in deciding the conversion. If a new page is introduced to user on day 1 or nighttime compared to day 10 or morning, the user could have different reaction. From looking at the minimum and maximum values for the timestamp, the data seemed to be collected only for 20 days. May be longer duration could give us more accurate results. The users for the new page might still be discovering additional useful features involved with the new page.**
# 
# **There are also some disadvantages to adding additional terms into the logistic regression model. For example, these additional terms can be correlated with one another. This would lead us to develop multicollinear model.**

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy varaibles.** Provide the statistical output as well as a written response to answer this question.

# In[97]:


df_countries = pd.read_csv('countries.csv')
df_countries.head()
df_2.head()


# In[98]:


df_new = df2.set_index('user_id').join(df_countries.set_index('user_id'), how='inner')
df_new.head()


# In[99]:


df_new['country'].value_counts()


# In[100]:


df_new[['CA','UK','US']] = pd.get_dummies(df_new['country'])
logmod_new = sm.Logit(df_new['converted'],df_new[['intercept','treatment','US','UK']])
results_new = logmod_new.fit()
results_new.summary()


# In[101]:


np.exp(results_new.params)


# In[102]:


1/_


# **1) If a user does not receive treatment, he/she is 1.05 more likely to be converted than if a user receives the treatment, holding all other variables constant.**
# 
# **2) If a user is from US, he/she is 1.04 more likely (4% more likely) to be converted than if a user is from Canada, holding all other variables constant.**
# 
# **3) If a user is from UK, he/she is 1.05 more likely (5% more likely) to be converted than if a user is from Canada, holding all other variables constant.**
# 
# **4) None of the variables(treatment, US, UK) are statistically significant since their p values are greater than type I error, 0.05 (or 5%).**
# 
# **5) The comparison of conversion for US and UK users, with the baseline Canadian users, the difference is very small (4% or 5%) to even be considered practically significant.**

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[103]:


from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
y, X = dmatrices('converted ~ treatment + US + UK', df_new, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"]= [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif["features"]=X.columns
vif
#Since VIF values for all the x variable are less than 10, there is no multicollineartiy present.


# In[104]:


df_new['US_treated']=df_new['US']*df_new['treatment']
df_new['UK_treated']=df_new['UK']*df_new['treatment']
df_new.head()


# In[105]:


logmod_new_1 = sm.Logit(df_new['converted'],df_new[['intercept','treatment','US','UK','US_treated','UK_treated']])
results_new_1 = logmod_new_1.fit()
results_new_1.summary()


# **The p-values for the new model are greater than Type I error, 0.05. Therefore, the x-variables associated with the new model are deemed statistically insignificant. The new model of considering a user received a new page and lived in a specific country failed to show statistical significance in predicting the conversion for the new page. **

# <a id='conclusions'></a>
# ## Finishing Up
# 
# > Congratulations!  You have reached the end of the A/B Test Results project!  This is the final project in Term 1.  You should be very proud of all you have accomplished!
# 
# > **Tip**: Once you are satisfied with your work here, check over your report to make sure that it is satisfies all the areas of the rubric (found on the project submission page at the end of the lesson). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.
# 
# 
# ## Directions to Submit
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[107]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])

