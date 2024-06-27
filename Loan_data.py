#!/usr/bin/env python
# coding: utf-8

# ### Importing Packages

# In[1]:


import numpy as np


# In[2]:


np.set_printoptions(suppress=True, linewidth=100, precision=2)


# ### Importing Data

# In[3]:


raw_data_np = np.genfromtxt('C:/Users/Admin/Desktop/Datasets/loan-data.csv',
                            delimiter=';',
                           skip_header=1,
                           autostrip=True)
raw_data_np


# ### Checking for Incomplete Data

# In[4]:


np.isnan(raw_data_np).sum()


# In[5]:


temporaray_fill = np.nanmax(raw_data_np) + 1
temporaray_mean = np.nanmean(raw_data_np, axis=0)


# In[6]:


temporaray_mean


# In[7]:


temporary_stats = np.array([np.nanmin(raw_data_np, axis=0),
                           temporaray_mean,
                           np.nanmax(raw_data_np, axis=0)])


# In[8]:


temporary_stats


# ### Splitting up Data

# #### Splitting up columns

# In[9]:


column_strings = np.argwhere(np.isnan(temporaray_mean)).squeeze()
column_strings


# In[10]:


column_numeric = np.argwhere(np.isnan(temporaray_mean) == False).squeeze()
column_numeric


# #### Reimporting the Dataset

# In[11]:


loan_data_strings = np.genfromtxt('C:/Users/Admin/Desktop/Datasets/loan-data.csv',
                                 delimiter=';',
                                 skip_header=1,
                                 autostrip=True,
                                 usecols=column_strings,
                                 dtype='str')
loan_data_strings


# In[12]:


loan_data_numeric = np.genfromtxt('C:/Users/Admin/Desktop/Datasets/loan-data.csv',
                                 delimiter=';',
                                 skip_header=1,
                                 autostrip=True,
                                 usecols=column_numeric,
                                 filling_values=temporaray_fill)
loan_data_numeric


# #### The Names of Columns

# In[13]:


header_full = np.genfromtxt('C:/Users/Admin/Desktop/Datasets/loan-data.csv',
                                 delimiter=';',
                                 skip_footer=raw_data_np.shape[0],
                                 autostrip=True,
                                 dtype='str')
header_full


# In[14]:


header_strings, header_numeric = header_full[column_strings], header_full[column_numeric]


# In[15]:


header_strings


# In[16]:


header_numeric


# ### Creating Checkpoints
# Places around the code where we store a copy of our dataset
# Avoid losing a lot of progress
# An extremely reliable approach when we need to clean or preprocess many parts of dataset
# A failsafe to rely on

# In[17]:


def checkpoint(file_name, checkpoint_header, checkpoint_data):
    np.savez(file_name, header=checkpoint_header, data=checkpoint_data)
    checkpoint_variable = np.load(file_name + '.npz')
    return checkpoint_variable


# In[18]:


checkpoint_test = checkpoint('checkpoint-test', header_strings, loan_data_strings)


# In[19]:


checkpoint_test['header']


# In[20]:


checkpoint_test['data']


# In[21]:


np.array_equal(checkpoint_test['data'], loan_data_strings)


# #### Manipulating String Columns

# In[22]:


header_strings


# In[23]:


header_strings[0] = 'issue_date'


# In[24]:


loan_data_strings[:,0]


# In[25]:


loan_data_strings[:,0] = np.chararray.strip(loan_data_strings[:,0], "-15")


# In[26]:


np.unique(loan_data_strings[:,0])


# In[27]:


months = np.array(['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])


# In[28]:


for i in range(13):
    loan_data_strings[:,0] = np.where(loan_data_strings[:,0] == months[i],
                                     i,
                                     loan_data_strings[:,0])


# In[29]:


np.unique(loan_data_strings[:,0])


# #### Loan Status

# In[30]:


header_strings


# In[31]:


np.unique(loan_data_strings[:,1])


# In[32]:


status_bad = np.array(['', 'Charged Off', 'Default', 'Late (31-120 days)'])


# In[33]:


loan_data_strings[:,1] = np.where(np.isin(loan_data_strings[:,1],
                                          status_bad),
                                 0,1)


# In[34]:


np.unique(loan_data_strings[:,1])


# #### Term

# In[35]:


header_strings


# In[36]:


np.unique(loan_data_strings[:,2])


# In[37]:


loan_data_strings[:,2] = np.chararray.strip(loan_data_strings[:,2], ' months')


# In[38]:


loan_data_strings[:,2]


# In[39]:


header_strings[2] = 'term_months'


# In[40]:


loan_data_strings[:,2] = np.where(loan_data_strings[:,2] == '',
                                 60,
                                 loan_data_strings[:,2])


# In[41]:


np.unique(loan_data_strings[:,2])


# #### Grade and Subgrade Values

# In[42]:


header_strings


# In[43]:


np.unique(loan_data_strings[:,3])


# In[44]:


np.unique(loan_data_strings[:,4])


# In[45]:


for i in np.unique(loan_data_strings[:,3])[1:]:
    loan_data_strings[:,4] = np.where((loan_data_strings[:,4] == '') & (loan_data_strings[:,3] == i),
                                     i + '5',
                                     loan_data_strings[:,4])


# In[46]:


np.unique(loan_data_strings[:,4], return_counts=True)


# In[47]:


loan_data_strings[:,4] = np.where(loan_data_strings[:,4] == '',
                                     'H1',
                                     loan_data_strings[:,4])


# In[48]:


np.unique(loan_data_strings[:,4])


# #### Removing Grade

# In[49]:


loan_data_strings = np.delete(loan_data_strings, 3, axis=1)


# In[50]:


loan_data_strings[:,3]


# In[51]:


header_strings = np.delete(header_strings, 3)


# In[52]:


header_strings


# #### Converting Subgrade

# In[53]:


np.unique(loan_data_strings[:,3])


# In[54]:


keys = list(np.unique(loan_data_strings[:,3]))
values = list(range(1, np.unique(loan_data_strings[:,3]).shape[0] + 1))
dict_sub_grade = dict(zip(keys, values))


# In[55]:


dict_sub_grade


# In[56]:


for i in np.unique(loan_data_strings[:,3]):
    loan_data_strings[:,3] = np.where(loan_data_strings[:,3] == i,
                                     dict_sub_grade[i],
                                     loan_data_strings[:,3])


# In[57]:


np.unique(loan_data_strings[:,3])


# #### Verification Status

# In[58]:


header_strings


# In[59]:


np.unique(loan_data_strings[:,4])


# In[60]:


loan_data_strings[:,4] = np.where((loan_data_strings[:,4] == '') | (loan_data_strings[:,4] == 'Not Verified'), 0,1)


# In[61]:


np.unique(loan_data_strings[:,4])


# #### URL

# In[62]:


header_strings


# In[63]:


np.unique(loan_data_strings[:,5])


# In[64]:


loan_data_strings[:,5] = np.chararray.strip(loan_data_strings[:,5], 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=')


# In[65]:


header_full


# In[66]:


loan_data_numeric[:,0].astype(dtype='int32')


# In[67]:


loan_data_strings[:,5].astype(dtype='int32')


# In[68]:


np.array_equal(loan_data_numeric[:,0].astype(dtype='int32'), loan_data_strings[:,5].astype(dtype='int32'))


# In[69]:


loan_data_strings = np.delete(loan_data_strings, 5, axis=1)
header_strings = np.delete(header_strings, 5)


# In[70]:


loan_data_strings[:,5]


# In[71]:


header_strings[5]


# #### State Address

# In[72]:


header_strings


# In[73]:


loan_data_strings[:,5]


# In[74]:


states_names, states_count = np.unique(loan_data_strings[:,5], return_counts=True)
states_count_sorted = np.argsort(-states_count)
states_names[states_count_sorted], states_count[states_count_sorted]


# In[75]:


loan_data_strings[:,5] = np.where(loan_data_strings[:,5] == '',
                                 0,
                                 loan_data_strings[:,5])


# In[76]:


states_west = np.array(['WA', 'OR','CA','NV','ID','MT', 'WY','UT','CO', 'AZ','NM','HI','AK'])
states_south = np.array(['TX','OK','AR','LA','MS','AL','TN','KY','FL','GA','SC','NC','VA','WV','MD','DE','DC'])
states_midwest = np.array(['ND','SD','NE','KS','MN','IA','MO','WI','IL','IN','MI','OH'])
states_east = np.array(['PA','NY','NJ','CT','MA','VT','NH','ME','RI'])


# In[77]:


loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_west), 1, loan_data_strings[:,5])
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_south), 2, loan_data_strings[:,5])
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_midwest), 3, loan_data_strings[:,5])
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_east), 4, loan_data_strings[:,5])


# In[78]:


np.unique(loan_data_strings[:,5])


# ### Manipulating Text Data to Numeric

# In[79]:


loan_data_strings


# In[80]:


loan_data_strings = loan_data_strings.astype('int32')


# In[81]:


loan_data_strings


# ### Checkpoint 1: Strings

# In[82]:


checkpoint_strings = checkpoint('Checkpoint-strings', header_strings, loan_data_strings)


# In[83]:


checkpoint_strings['header']


# In[84]:


checkpoint_strings['data']


# In[85]:


np.array_equal(loan_data_strings, checkpoint_strings['data'])


# ### Manipulating Numeric Columns 

# In[86]:


loan_data_numeric


# In[87]:


np.isnan(loan_data_numeric).sum()


# #### Substitute "Filler" Values 

# In[88]:


header_numeric


# #### ID

# In[89]:


temporaray_fill


# In[90]:


np.isin(loan_data_numeric[:,0], temporaray_fill).sum()


# #### Temporary Stats

# In[91]:


temporary_stats[:, column_numeric]


# #### Funded Amnt

# In[92]:


loan_data_numeric[:,2]


# In[93]:


loan_data_numeric[:,2] = np.where(loan_data_numeric[:,5] == temporaray_fill,
                                 temporary_stats[0, column_numeric[2]],
                                 loan_data_numeric[:,2])
loan_data_numeric[:,2]


# In[94]:


temporary_stats[0, column_numeric[2]]


# #### Loaned Amnt, Interest Rate, Total Paymnt, Installment

# In[95]:


header_numeric


# In[96]:


for i in [1,3,4,5]:
    loan_data_numeric[:,i] = np.where(loan_data_numeric[:,i] == temporaray_fill,
                                     temporary_stats[2, column_numeric[i]],
                                     loan_data_numeric[:,i])


# In[97]:


loan_data_numeric


# ### Currency Change

# #### The Exchange Rate

# In[98]:


EUR_USD = np.genfromtxt("EUR-USD.csv", delimiter = ',', autostrip = True, skip_header = 1, usecols = 3)
EUR_USD


# In[99]:


loan_data_strings[:,0]


# In[100]:


exchange_rate = loan_data_strings[:,0]

for i in range(1,13):
    exchange_rate = np.where(exchange_rate == i,
                             EUR_USD[i-1],
                             exchange_rate)    

exchange_rate = np.where(exchange_rate == 0,
                         np.mean(EUR_USD),
                         exchange_rate)

exchange_rate


# In[101]:


exchange_rate.shape


# In[102]:


loan_data_numeric.shape


# In[103]:


exchange_rate = np.reshape(exchange_rate, (10000,1))


# In[104]:


loan_data_numeric = np.hstack((loan_data_numeric, exchange_rate))


# In[105]:


header_numeric = np.concatenate((header_numeric, np.array(['exchange_rate'])))
header_numeric


# #### From USD to EUR

# In[106]:


header_numeric


# In[107]:


columns_dollar = np.array([1,2,4,5])


# In[108]:


loan_data_numeric[:,6]


# In[109]:


for i in columns_dollar:
    loan_data_numeric = np.hstack((loan_data_numeric, np.reshape(loan_data_numeric[:,i] / loan_data_numeric[:,6], (10000,1))))


# In[110]:


loan_data_numeric.shape


# In[111]:


loan_data_numeric


# #### Expanding the header

# In[112]:


header_additional = np.array([column_name + '_EUR' for column_name in header_numeric[columns_dollar]])


# In[113]:


header_additional


# In[114]:


header_numeric = np.concatenate((header_numeric, header_additional))


# In[115]:


header_numeric


# In[116]:


header_numeric[columns_dollar] = np.array([column_name + '_USD' for column_name in header_numeric[columns_dollar]])


# In[117]:


header_numeric


# In[118]:


columns_index_order = [0,1,7,2,8,3,4,9,5,10,6]


# In[119]:


header_numeric = header_numeric[columns_index_order]


# In[120]:


loan_data_numeric


# In[121]:


loan_data_numeric = loan_data_numeric[:,columns_index_order]


# ### Interest Rate

# In[122]:


header_numeric


# In[123]:


loan_data_numeric[:,5]


# In[124]:


loan_data_numeric[:,5] = loan_data_numeric[:,5]/100


# In[125]:


loan_data_numeric[:,5]


# ### Checkpoint 2: Numeric

# In[126]:


checkpoint_numeric = checkpoint("Checkpoint-Numeric", header_numeric, loan_data_numeric)


# In[127]:


checkpoint_numeric['header'], checkpoint_numeric['data']


# ### Creating the "Complete" Dataset

# In[128]:


checkpoint_strings['data'].shape


# In[129]:


checkpoint_numeric['data'].shape


# In[130]:


loan_data = np.hstack((checkpoint_numeric['data'], checkpoint_strings['data']))


# In[131]:


loan_data


# In[132]:


np.isnan(loan_data).sum()


# In[133]:


header_full = np.concatenate((checkpoint_numeric['header'], checkpoint_strings['header']))


# ### Sorting the Dataset 

# In[134]:


loan_data = loan_data[np.argsort(loan_data[:,0])]


# In[135]:


loan_data


# In[136]:


np.argsort(loan_data[:,0])


# ### Storing the Dataset

# In[139]:


loan_data = np.vstack((header_full, loan_data))


# In[140]:


np.savetxt('loan-data-preprocessed.csv',
           loan_data,
           fmt='%s',
           delimiter=',')


# In[ ]:




