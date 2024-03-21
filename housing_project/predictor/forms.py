from django import forms

class HouseFeaturesForm(forms.Form):
    CRIM = forms.FloatField(label='Per capita crime rate by town')
    ZN = forms.FloatField(label='Proportion of residential land zoned for lots over 25,000 sq.ft.')
    INDUS = forms.FloatField(label='Proportion of non-retail business acres per town')
    CHAS = forms.ChoiceField(label='Charles River dummy variable (1 if tract bounds river, 0 otherwise)', choices=[(0, 'No'), (1, 'Yes')], widget=forms.RadioSelect)
    NOX = forms.FloatField(label='Nitric oxides concentration (parts per 10 million)')
    RM = forms.FloatField(label='Average number of rooms per dwelling')
    AGE = forms.FloatField(label='Proportion of owner-occupied units built prior to 1940')
    DIS = forms.FloatField(label='Weighted distances to five Boston employment centres')
    RAD = forms.IntegerField(label='Index of accessibility to radial highways')
    TAX = forms.FloatField(label='Full-value property-tax rate per $10,000')
    PTRATIO = forms.FloatField(label='Pupil-teacher ratio by town')
    B = forms.FloatField(label='1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town')
    LSTAT = forms.FloatField(label='% Lower status of the population')