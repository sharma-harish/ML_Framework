import category_encoders as ce

NOM_ENCODER = {
    'OneHotEncoder' : ce.OneHotEncoder(cols=['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'])
}