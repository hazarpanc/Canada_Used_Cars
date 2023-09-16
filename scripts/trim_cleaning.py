import re
import logging
import pandas as pd

logging.basicConfig(
    filename='./logs/preprocessing_results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def remove_extra_spaces(input_string):
    return re.sub(r'\s+', ' ', input_string)

def remove_second_occurrence(s):
    words = s.split(' ')
    first_word = words[0]
    first_occurrence = True
    for i, word in enumerate(words):
        if word == first_word:
            if first_occurrence:
                first_occurrence = False
            else:
                words[i] = ''
    return ' '.join(words)

def remove_duplicate_words(input_string):
    words = input_string.split()
    seen_words = set()
    output_words = []

    for word in words:
        if word not in seen_words:
            output_words.append(word)
            seen_words.add(word)

    return ' '.join(output_words)


def correct_string(input_str, mapping):
    # Convert the input string to lowercase for case-insensitive matching
    lower_input_str = input_str.lower()
    
    # Check if the input string is in the mapping, and return the corresponding value if found
    if lower_input_str in mapping:
        return mapping[lower_input_str]
    
    # If the input string is not found in the mapping, return the original input string
    return input_str


def extract_info_from_trim(df):
    """
    Update specific fields in a DataFrame based on the presence of substrings
    in the 'trim' column.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
                                  
    Returns:
    pandas.DataFrame: DataFrame with updated fields.
    """
    
    # Define the list of updates
    update_list = [
        ('bodytype', 'sedan', 'sedan'),
        ('bodytype', 'coupe', 'coupe'),
        ('bodytype', 'hatchback', 'hatchback'),
        ('bodytype', 'hatch', 'hatchback'),
        ('bodytype', 'hayon', 'hatchback'),
        ('bodytype', '2 portes', 'coupe'),
        ('bodytype', 'convertible', 'convertible'),
        ('bodytype', 'cabriolet', 'cabriolet'),
        #('bodytype', 'wagon', ''),
        ('fueltype', 'diesel', 'diesel'),
        ('fueltype', 'bluetec', 'diesel'),
        ('fueltype', 'tdi', 'diesel'), # left a space to avoid words
        ('fueltype', 'hybrid', 'hybrid'),
        ('fueltype', 'hybride', 'hybrid'),
        ('fueltype', 'hybride branchable', 'hybrid'),
        ('fueltype', 'vehicule electrique', 'electric'),
        ('fueltype', 'electric motor', 'electric'),
        #('fueltype', 'electrique', 'electric'),
        #('fueltype', 'electrique', 'electric'),
        ('drivetrain', '4 roues motrices', 'AWD'),
        ('drivetrain', 'all wheel drive', 'AWD'),
        ('drivetrain', 'quattro', 'AWD'),
        ('drivetrain', '4 matic', 'AWD'),
        ('drivetrain', '4matic', 'AWD'),
        ('drivetrain', 'awd', 'AWD'),
        ('drivetrain', '4wd', 'AWD'),
        ('drivetrain', 'dual motor', 'AWD'),
        ('drivetrain', '4x4', 'AWD'),
        ('drivetrain', 'all4', 'AWD'),
        ('drivetrain', '2wd', 'FWD'),
        ('drivetrain', '2rm', 'FWD'),
        ('drivetrain', 'traction intégrale', 'AWD'),
        ('drivetrain', 'traction integrale', 'AWD'),
        ('drivetrain', 'xdrive', 'AWD'),
        ('drivetrain', 'quattro', 'AWD'),
        ('drivetrain', '4motion', 'AWD'),
        ('drivetrain', 'rwd', 'RWD'),
        ('drivetrain', 'fwd', 'FWD'),
        ('transmission', 'cvt', 'auto'),
        ('transmission', 'ivt', 'auto'),
        ('transmission', 'dct', 'auto'),
        ('transmission', 'pdk', 'auto'),
        ('transmission', 'dsg', 'auto'),
        ('transmission', 'ivt', 'auto'),
        ('transmission', 's-tronic', 'auto'),
        ('transmission', 'tiptronic', 'auto'),
        ('transmission', '6mt', 'manual'),
        ('transmission', 'manuelle', 'manual'),
        ('transmission', 'manual', 'manual'),
        ('transmission', 'manuel', 'manual'),
    ]

    # Loop over the list of tuples
    for column, substring, value in update_list:
        # Create a boolean mask where True indicates that the 'trim' column
        # contains the current substring (case-insensitive)
        mask = df['trim'].str.contains(substring, case=False, na=False)

        # Update the specified column based on the mask
        df.loc[mask, column] = value
    
    # Return the updated DataFrame
    return df


def clean_trim(trim, add_unwanted_words=None):
    """
    To be used in process trim function
    Clean and standardize the trim of a car given as a string.

    This function takes a trim value as input and performs several transformations to clean and standardize it.

    Parameters:
    - trim (str): The original trim string to be cleaned.
    - add_unwanted_words (list, optional): A list of additional words to be removed from the trim string.
      Defaults to None.

    Returns:
    - str: The cleaned and standardized trim string.

    Notes:
    - The function performs the following transformations:
        1. Converts the trim to lowercase.
        2. Removes any extra spaces and leading/trailing spaces.
        3. Removes common invalid characters like '!', '*', '/', '+'.
        4. Removes certain unwanted words (usually marketing words) from the trim.
        5. Removes second occurrences of the first word in the trim.
        6. The `add_unwanted_words` parameter allows you to specify additional words to be removed.
    """
    trim = str(trim).lower()
        
    #Truncate the string after the first occurrence of the word
    def truncate_after_words(s, words):
        for word in words:
            if word in s:
                s = s.split(word, 1)[0].strip()
                break
        return s    
    
    words = ["|",","," - "," w/","with","avec","~","("]
    trim = truncate_after_words(trim, words)
    
    # Remove invalid characters 
    characters_to_remove = ['!', '*', '/', '+', '~', '<', '>', '"', '®', '™',"\\",';',"&"]
    for char in characters_to_remove:
        trim = trim.replace(char, " ")
    
    # Replace problematic chars
    trim = trim.replace("'", "ft")
    
    # Remove extra spaces
    trim = remove_extra_spaces(trim)
    trim = trim.strip()
    
    ### STEP 2: TRANSLATE
    # Fix French variations of model names
    # Dictionary mapping French variations to English translations
    def translate_string(s, translations, sort_translations_keys=False):
        """
        Translates parts of a string based on a given mapping.
        """
        # Sort the dictionary by length of key (descending)
        if sort_translations_keys:
            translations = {k: v for k, v in sorted(translations.items(), key=lambda item: len(item[0]), reverse=True)}

        # Create a combined pattern for all keys
        pattern = re.compile("|".join(map(re.escape, translations.keys())))

        # Use the pattern to replace occurrences in the string
        return pattern.sub(lambda m: translations[m.group(0)], s)
    
    trim_translation_mapping = {
        "e anniversaire" : "th anniversary",
        "vision climat" : "climate vision",
        "gr. remorquage" : "towing package",
        "gr.remorquage" : "towing package",
        "disponibilité limited" : "limited",
        "groupe remorquage" : "towing package",
        "hybride rechargeable": "plug-in hybrid",
        "hybride branchable" : "plug-in hybrid",
        "plug in" : "plug-in",
        "technologie" : "technology",
        "privilège" : "privilege",
        'tourisme' :'touring',
        'preffered' : "preferred",
        "privilégié" : "preferred",
        'executif':'executive',
        "électriques" : "electric",
        "électrique" : "electric",
        "electriques" : "electric",
        "electrique" : "electric",
        "automique" : "atomic",
        "autoplot" : "autopilot",
        "autonomie":"autopilot",
        "portes": "door",
        "confort":"comfort",
        "édition":"edition",
        'caligraphy':'calligraphy',
        "s line":"s-line",
        "fsport" : "f-sport",
        "f sport" : "f-sport",
        "luxe":"luxury",
        "hybride": "hybrid",
        "hayon": "hatchback",
        "berline": "sedan",
        "coupé" : "coupe",
        "cabriolet" : "convertible",
        "décapotable" : "convertible",
        "limitée" : "limited",
        "limité" : "limited",
        "sélect" :"select",
        "allongé" : "crew cab 143.5",
        "prem plus" : "premium plus",
        "platine" : "platinum"
    }
    
    trim = translate_string(trim, trim_translation_mapping)
    
    ### STEP 3: REMOVE UNWANTED WORDS
    
    if add_unwanted_words is None:
        add_unwanted_words = []
        
    # Remove unwanted words (usually marketing words) from the trim
    complex_words_to_remove = ['( 2 ans inclus)', '$500 finance incentive','0.99%', '4.49%', "3.99%","2.99%","3.39%","4.49%","5.19%","1.99%","5.53%",'gr.electric', 'magstoit.ouvrantsieges.chauff', 
                               'magssieges.chauffbluetooth', 'navicuirtoit.ouvrant', 'gr.electrique', 'magscam.reculapple.', 'b. h',
                               "taux à partir de", "à partir de",'18po-mags','o.a.','like-new',"5.99",'drivr.asst',
                               "gr.electrique", "magscam.reculapple.", 'gr-electric', "4-dr", "5-dr", 'b. h', 'h- r', '1 an', 'o.a.c.','o.a.',
                               'b h', "®", "à", "b&"] 
    
    simple_words_to_remove = [
                    # Complex                    
                    'ask us how we can find you a similar vehicle', 'premjamais accidentégarantie','10ans200000km', '7 ans 160km','10ans 200000km','160 000km',
                    'compatible apple carplay et android auto','drives great','runs great', 'sport utility vehicle', 'sport utility',
                    'great deal', 'buy now', 'hurry before it sells out', 'buy now before it sells out', 'advanced driving assistant', 
                    'groupe electrique complet', 'en attente dftapprobation', 'as isyou certifyyou save', 'all credits', 
                    'financement disponible', 'gr-électrique -ouvrant', 'we finance', 'we approve', 'all credit', 'heated steering wheel', 
                    '5 years 160,000km wrt', 'disponibilité limited', 'compatible et android', 'disponibilité limitée', 'disponibilité limitéd', 
                    'full service history', 'seul','un', 'gr-électrique', 'sièges volant chauff', 'demarreur a distance', 'come and test drive', 
                    'excellent condition', 'convenience package', 'ambient light drive', 'trades','welcome', '1 seul propriétaire', 'spring sale event on now', 
                    'spring sale', 'traction intégrale', 'sulev south africa', 'garantie prolongée', 'vitres électriques', 'aide à la conduite', 
                    'entièrement équipé', 'warranty included', '5 years 160,000km', 'wireless charging', 'jantes en alliage', 'sièges chauffants', 
                    'sieges chauffants', 'seul proprietaire', 'groupe électrique', 'camion de travail', 'groupe electrique', 'panoramic sunroof', 
                    'heads up display', 'air conditioning', 'sport appearance', 'sieges chauffant', 'jamais accidenté', 'jamais accidente', 
                    'volant chauffant', 'sièges chauffant', '4 roues motrices', '7-passenger', '7-passagers', 'app-connect', '8-pass', 
                    '7-pass', '4-door', '5-door', 's-tronic', '8 pneus', '35th annversary edton', 'collision alert', 'collision detection', 
                    '6-speed', '5 speed', '6 speed', 'like new','from oa', 'l o', 'less than 40','sliding doors',
                    '4 door', '3 door', '2 door', 'head-up display', 'head up display', 'all wheel drive', 'available as is', 'all-wheel drive', 
                    'convenience pkg', 'awdpneus inclus', 'siege chauffant', 'bas kilométrage', 'elect cam recul', 'caméra de recul', 'camera de recul', 
                    'bas kilometrage', 'bas kilo', 'ventilés', 'groupe electric', 'cruise control', 'remote starter', 'headup display', 
                    'apple car play', 'steering wheel', 'w-wing spoiler', 'low kilometers', 'low kilometres', '1 propriétaire', '1 proprietaire', 
                    'sxt stow nftgo', 'whonda sensing', 'banc chauffant', 'ens commodités', 'taux partir de', 'panoramic roof', 'incoming unit', 
                    'adv key', 'all service records', 'full service records', 'service records','services records','bien entretenu',
                    'systeme de', 'driver assist', '5 door',"18 wheels",'21 wheels','7 seater','car play', 'groupe comfort',
                    'accident free', 'backup camera', 'harman kardon', 'as-is special', 'keyless entry', 'air condition', 'apple carplay', 
                    'compatible et', 'north america', 'ambient light', 'adaptive crse', 'groupe valeur', '160,000km wrt', 'no luxury tax', 
                    'ens commodité', 'ens commodite', 'gr electrique', 'écran tactile', 'gr électrique', '---vitre elec', 
                    'w-li battery', 'clean carfax', 'heated seats', 'fully loaded', 'no accidents', 'single owner', 'remote start', 
                    '7 passengers', '7-seater', 'alloy wheels', 'sale pending', 'blowout sale', 'south africa', 'used vehicle', 'wing spoiler', 
                    'sound system', 'sieges promo', 'to your door', 'toit ouvrant', 'groupe élect', 'volant chauf', 'série de bmw', 'app connect', 
                    'low mileage', 'new arrival', 'no accident', 'cloth seats', '8 passenger', '8 passagers', '7 passenger', '7 passagers', '7 passager', 
                    'lane assist','lane departure', 'ltd avail', 'fresh trade', 'coming soon', 'park assist', 'série de bm', 'easy trades', '6 passagers', 
                    'sxt stowngo', 'valeur plus', 'angle morts', 'gr electric', 'air ventilé', 'bas millage', 'blue tooth', 
                    'backup cam', 'push start', 'test drive', '-ltd avail', 'blind spot', '&#174;', '200 000 km', 'comme neuf', 'ans inclus', 
                    'car fax', '7 ans', "financement a",'top of line', 'cold weather','vitre electric', '7 sts',
                    '7 ans160km', 'vitre elec', 'off lease', 'one owner', 'w adv key', 'ltd avail', '200 000km', '1 proprio', 'new tires', 
                    'new brakes', 'all around', 'as traded', 'in hearst', 'low kmfts', 'rec siège', 'rec siege', '-18ftft', '60 mois',
                    'rég adapt', 'voiture à', 'cam recul', '18 pouces', '19 pouces', '20 pouces', '2 portes', '12m-20', '3 portes', '4 portes', 
                    '5 portes', '6 pass', '8 pass', '7 pass', 'low km', 'low kms', '3 year', 'w leds', '1 owne','just in',
                    'w heat', 'bas km', '10 ans', '4 cyl', 'as is', '4 rm',"a c",'24 months','24 mths', '12 months', '12 mths','36 months',
                    # Very simple
                    "must see", 'electric', 'elect', 'elec', 'model', 'convertible',"coupe","1",
                    "carplayandroidauto", "bluetoothcamera", "carplayandroid","androidauto", "certification", "entertainment", 
                    "climatisation", "personnalisée", "transmission", "automatique", "certifiable", 'grade','lanewatch','launch',
                    'économique','economique', 'compatible','suspension','you','certify','save','version','led','voiture',
                    "8-passenger", "touchscreen", "convenience", "climatiseur", "financement", 'sieges','promo','siéges','bancs','banc','toyota',
                    "panoramique", "7passagers", "commodités", "régulateur", "navigation", "commodites", "pneus","inclus", 'proprietaire','propriétaire',
                    "panocaméra", "hatchback", "bluetooth", "available", "automatic", "blindspot", "tiptronic", "démarreur", 
                    "excellent", "condition", "burmester", "familiale", "delivered", "interieur", "adaptatif", "intégrale", "commodité", 
                    "commodite", "climatise", "certified", "demarreur", "panoramic", "manuelle", "bluetoot", "warranty", "incoming", 
                    "interior", "adaptive", "pkglexus", "extended", "wireless", "ensemble", "vdpurlen", 'ventilated', 'noire','reduced',
                    "accident", "approval", "delivery", "arrivage", "rapporte", "dfthiver", "distance", "steering", "garantie", "certifie", 
                    "certifié", "inspecté", "amélioré", "panoroof", "moonroof", "2portes", "berline", "upgrade", "sunroof", "1-owner", 
                    "massage", "leather", "android", "edition", "reserve", "keyless", "carplay", "spoiler", "special", "arrived", 'climatisé',
                    "arrival", "financing","finance", "navigat", "vitesse", "located", "pending", "spécial", "complet", "ouvrant", "réserve", 
                    "reverse", "ventilé", "alertes", "alerte", 'alert', "diesel", "carfax", "loaded", "alloys", "cruise", 'globale','ready',
                    'brake','rotors','prem','seat','winter','tires','drivr','bluethoot','brand', 'lift', 'wheel', 'tire','regul','white',
                    "360cam", "safety", "backup", "remote", "recent", "modèle", "credit", "chauff", "jamais", "heated", "heat",'rec',
                    "angles", "volant", "landed", "minuit", "manual", "recule", "manuel", "caméra", "camera", "vitres", "sièges","siège", "sedan", 
                    "hayon", "cloth", "hatch", "6pass", "7pass", "as-is", "local", "400hp", "567hp", "650hp", "237hp", "audio", 'comes', "from",
                    "trade", "sound", "adapt", "apple", "power", "clean", "owner", "aucun", "owned", "hurry", "seats", "chauf", "group",'hiver', 
                    "other", "elect", "recul", "vendu", "siege", "écran", "camer", "boîte", "6spd", "auto", "bose", "roof", "500km", "500k",
                    "with", "mint", "rare", "4cyl", "very", "wing", "crse", "rear", "just", "sold", "only", "mois", "avec", "deal", "sale", 'tres','deux',
                    "leds", 'pan', '1ère', 'jbl','this','new','front','brakes','lexus','cooled','18po','10ans','7ans','chau','taux','fixe','dispo',
                    "easy", "clim", "2016", "2018", "2013", "2017", "2012", "2019", "2020", "2022", "2021", "2014", "2015", "2023", "2024", 'and',
                    "2011", "2010", "vent", "navi", "cuir", "mags", "toit", 'sortie',"voie", "pano", "one","awd", "4wd", "4x4", "rwd", "fwd", "2wd", "2rm", 
                    "4rm", "4dr", "2dr", "3dr", "5dr", "8sp", "7sp", "sdn", "cpe", "suv", "usb", "aux", "cvt", "hud", "pkg", "air", "6sp", 'pre','amp','oac',
                    "6mt", "mag", "360", "at4", "aut", "man", "dsg", "nav", "gps", "dvd", "cam", "cpo", "wow","woww",'htd', "4d", "hb","oa",
                    "ac", "ca", "ba", "mt", "ti", "ta", "bm", "at", "gr","w-","w"]+ add_unwanted_words
    
    # Sort the words by length in descending order
    #multiple_words_to_remove.sort(key=len, reverse=True)
    #single_words_to_remove.sort(key=len, reverse=True)
    
    
    for word in complex_words_to_remove:
        trim = trim.replace(word, '')

    # Create a pattern that matches any of the words in words_to_remove
    #complex_words_pattern = r'(?<=^|\W)(' + '|'.join(re.escape(word) for word in complex_words_to_remove) + r')(?=\W|$)'
    simple_words_pattern = r'\b(?:' + '|'.join(re.escape(word) for word in simple_words_to_remove) + r')\b'

    # Use re.sub to replace all occurrences of any word in the pattern with an empty string
    #trim = re.sub(complex_words_pattern, '', trim)
    trim = re.sub(simple_words_pattern, '', trim)
    
    ### STEP 4: FINAL OPERATIONS
    
    # Remove extra spaces
    trim = remove_extra_spaces(trim)
    trim = trim.strip()
    
    # Remove second occurences of the first word
    trim = remove_duplicate_words(trim)
    
    # Remove invalid characters such as dash, dot from the end
    trim = trim.rstrip("-")
    trim = trim.rstrip(".")
    trim = trim.rstrip(" w")
    trim = trim.rstrip(" w-")
    trim = trim.rstrip(" &")
        
    # Remove extra spaces
    trim = remove_extra_spaces(trim)
    trim = trim.strip()
        
    return trim




def validate_trim(trim, add_invalid_trims=None):
    """
    Validate the trim of a car given as string.
    
    This function takes a trim value as input and checks if it is valid. If not, replaces it with unknown.

    Parameters:
    - trim (str): The original trim string to be validated.
    - add_invalid_trims (list): An optional list of additional invalid trims.

    Returns:
    - str: The validated and standardized trim string.
    """
    
    # If no additional invalid trims are provided, default to an empty list
    if add_invalid_trims is None:
        add_invalid_trims = []
    
    # This checks the whole trim string and if it equals any of the words below, it is replaced with unknown
    invalid_trims = ["nan","-","&","|",".","low","no","w","*","","#", "%","(",
                     "sedan", "cpe", "premium package", "manual", "wgn", "360", "air", "&", "bm","bt","mt","i",
                         "premium essential", "prem pkg", "1 owner", "one owner","1","2","3","4","5","h",
                           "accident free", "headup display", "premium", "clean", "system", "lo",
                            '-free', 'et','en','doors','car',"pre",'vehicle',"sun","range",
                           "bluetooth","sky view roof","leather","loaded", "incoming", "at","and",
                     "- premium essential", "- premium enhanced", ". 19", ",", ".5 gs", ".5 gx", "- t6",
                          "hatchback","série de bm"] + add_invalid_trims
    if trim in invalid_trims:
        return 'unknown'

    
    # Define the function to check if trim contains any words that proves it is not a valid trim name
    def check_redflag_words_in_string(input_string):
        '''
        Returns true if the string contains invalid words
        '''
        words_to_check = ["low kilometers","low kilometres","familiale","manuelle","certified","delivered",
                     "excellent","automatique","apple","local","camera","nouvel","backup",'extra',
                     "pano","panoramic","remote","rear","recent","incoming",'modèle', 'nav', 'commodité', 
                     'just',"arrived","arrival","sold",'ensemble','vdpurlen','édition','cuir','navi', 'panoroof',
                     'toit','only','power','ac', 'carfax','clean',"owner", "accident", "finance",
                         "financement","mois", "commodités", "certification","credit","approval","avec","aucun",
                          "bluetooth","headup","owned","delivery","deal","hurry","chauff","arrivage",
                          "navigat","jamais","heated","seats","moonroof","angles","leather","rapporte","garantie",
                          "recul","vitesse","interieur","located","touchscreen","sale","dfthiver","volant","chauf",
                          "ans inclus",".rec","morts","#", "%"
                         ]
        
        pattern = re.compile('|'.join(re.escape(word) for word in words_to_check))
        return pattern.search(input_string)
    
    # Check if trim contains any words that proves it is not a valid trim name
    if check_redflag_words_in_string(trim):
        return "unknown"
    
        
    # Return the validated and standardized trim
    return trim



def get_sorted_trims_above_threshold(df, make, threshold):
    """
    Get a sorted list of vehicle trims above a given threshold for a specific make.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        make (str): The make of the vehicles to consider.
        threshold (int): The minimum count threshold for a trim to be included.
        
    Returns:
        list: A sorted list of vehicle trims above the specified threshold for the given make.
    """
    
    # Filter the DataFrame to get only rows with the specified make
    make_data = df[df['make'] == make]
    
    # Calculate trim value counts and filter by the threshold
    trim_counts = make_data.trim.value_counts()
    #trims_above_threshold = list(trim_counts[trim_counts >= threshold].index.unique())
    trims_above_threshold = trim_counts[trim_counts >= threshold].index.drop('unknown', errors='ignore')
    
    
    # Remove the 'unknown' trim (if it exists)
    if 'unknown' in trims_above_threshold:
        trims_above_threshold.remove('unknown')
        
    # Remove trims that are 1 or 2 characters long from the list
    #trims_above_threshold = [item for item in trims_above_threshold if len(item) > 2]
    valid_trims = trims_above_threshold[trims_above_threshold.str.len() > 2]
    
    # Sort trims above the threshold based on their length in descending order
    #sorted_trims_above_threshold = sorted(trims_above_threshold, key=lambda x: len(x), reverse=True)
    
    # Convert valid_trims to a pandas Series
    valid_trims_series = pd.Series(valid_trims)

    # Sort the Series based on the length of its string values
    sorted_trims_series = valid_trims_series.iloc[valid_trims_series.str.len().sort_values(ascending=False).index]

    # Convert the sorted Series back to a list
    sorted_trims_above_threshold = sorted_trims_series.tolist()
    
    return sorted_trims_above_threshold



def update_unknown_trim(row, valid_trims):
    """
    If the trim is unknown, the function tries to find a valid trim in the trim_backup.

    Args:
        row (pd.Series): A row of the DataFrame.
        valid_trims (list of str): A list of valid trim values.

    Returns:
        pd.Series: The row with the 'trim' column possibly updated.
    """
    # If trim is not unknown, just return the row as is
    if row["trim"] != "unknown":
        return row
    
    # Use next with a generator expression for more efficient searching
    found_trim = next((trim for trim in valid_trims if trim in row["trim_backup"]), None)
    
    if found_trim:
        row["trim"] = found_trim
    
    # If no valid trim is found, return the row as is
    return row




def process_trim_by_make(df, make, trim_correction_mapping=None, valid_trims=None, add_unwanted_words=None, add_invalid_trims=None, unknown_threshold=4):
    """
    Cleans and processes the 'trim' column of a DataFrame for a specific car make.
    
    Args:
    df (pd.DataFrame): The input DataFrame.
    make (str): The make of the car to process.
    trim_correction_mapping (dict): A mapping of incorrect trim values to correct ones.
    valid_trims (list of str): A list of valid trim values.

    Returns:
    pd.DataFrame: The DataFrame with the 'trim' column processed.
    """
    print(f"--Processing {make}")
    # Create a boolean mask for rows where 'make' matches the input make
    mask = df['make'] == make
    
    # Clean 'trim' column for the specified make        
    df.loc[mask, 'trim'] = df.loc[mask, 'trim'].apply(clean_trim, add_unwanted_words=add_unwanted_words)
    
    # Apply trim correction mapping
    if trim_correction_mapping:
        df.loc[mask, 'trim'] = df.loc[mask, 'trim'].replace(trim_correction_mapping)
    
    # Check any invalid trims and replace them with unknown
    df.loc[mask, 'trim'] = df.loc[mask, 'trim'].apply(validate_trim, add_invalid_trims=add_invalid_trims)

    # Give entries with unknown trims a second chance
    # Try to find the right trims for them
    if valid_trims == "auto":
        # Use function to get valid trims
        valid_trims = get_sorted_trims_above_threshold(df, make, unknown_threshold)
        
        # Update trims
        df.loc[mask] = df.loc[mask].apply(update_unknown_trim, args=(valid_trims,), axis=1)
    elif valid_trims:
        df.loc[mask] = df.loc[mask].apply(update_unknown_trim, args=(valid_trims,), axis=1)
    
    return df





def correct_model_and_trim(df, make, model_correction_dict):
    """
    Update the 'model' and 'trim' of cars of a specific make in a DataFrame based on a correction dictionary.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing car data.
        make (str): The car make to which the corrections should be applied.
        make_correction_dict (dict): A dictionary containing the incorrect models as keys
                                     and the correct models and trims as values.

    Returns:
        pd.DataFrame: The updated DataFrame.
    """

    # Create a mask for the specific make
    make_mask = (df['make'] == make.lower())
    
    # Apply the corrections only on rows of the DataFrame that correspond to the specific make
    for incorrect_model, correction in model_correction_dict.items():
        # Create a mask for the incorrect model
        model_mask = df.loc[make_mask, 'model'].str.contains(incorrect_model, na=False)
        
        # Correct the model and trim using the masks
        df.loc[make_mask & model_mask, 'model'] = correction['model']
        df.loc[make_mask & model_mask, 'trim'] = correction['trim']

    return df


def process_trim(df, min_occurrences=5, combine_with_modelname=True):
    """
    Clean and process the 'trim' column of the provided DataFrame.

    This function applies a series of transformations to clean the 'trim' column and reduce the number of unique values.
    The same trim names can exist for different makes and this can affect the performance of the ML model.
    This fuction combines the trim name with the manufacturer name to overcome this issue.
    Removes invalid values, keeps only common trims that occur multiple times in data, and fills missing values with 'nan'.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the 'trim' column to be processed.
    - trim_threshold (int): The minimum number of occurrences required for a trim to be considered common.

    Returns:
    - pandas.DataFrame: The DataFrame with the 'trim' column modified and processed.

    """
    # Convert to string, then to lowercase
    df['trim'] = df['trim'].astype(str).str.lower()

    # Update bodytype, fueltype, transmission information of cars based on the trim column
    df = extract_info_from_trim(df)    

    # Create a backup of trim
    df['trim_backup'] = df['trim']
    
    print("Trims: Starting to process individual makes")
    
    ##
    ## PROCESS DIFFERENT MAKES - START
    ##
    
    # BMW
    bmw_trim_correction_mapping = {
        'competition coupe m' : 'competition m coupe',
        'm-sport' : 'm sport',
        'msport' : 'm sport',
        'm competition' : 'competition m',
        'm-competition' : 'competition m',
        "conv":"convertible",
    }
    invalid_bmw_trims = ["bmw","prem"]
    words_to_remove = ["xdrive"]
    df = process_trim_by_make(df, make='bmw', trim_correction_mapping=bmw_trim_correction_mapping, valid_trims="auto", add_unwanted_words=words_to_remove, add_invalid_trims=invalid_bmw_trims)
    
    
    # Toyota
    toyota_trim_correction_mapping = {
        'limited pkg' : 'limited',
        'limited hybrid' : 'hybrid limited',
        'xle hybrid' : 'hybrid xle',
        'xle hybride' : 'hybrid xle',
        'le hybrid' : 'hybrid le',
        'xse hybrid' : 'hybrid xse',
        'trd off-road' : 'trd off road',
        'hybride' : 'hybrid',
        'se 6m' : 'se',
        'se model' : 'se',
        '-e' : 'hybrid',
        'tech':'technology'
    }

    invalid_toyota_trims = ["wgn", "wgn v6"]
    df = process_trim_by_make(df, "toyota", toyota_trim_correction_mapping, None, None, invalid_toyota_trims)
    
    # Audi
    audi_trim_correction_mapping = {
        'progressiv 2.0 tfsi' : '2.0 tfsi progressiv',
        'technik 3.0 tfsi' : '3.0 tfsi technik',
        'progressiv 3.0 tfsi' : '3.0 tfsi progressiv',
        'progressiv 2.0 tfsi' : '2.0 tfsi progressiv',
        'technik 2.0 tfsi' : '2.0 tfsi technik',
        'komfort 2.0 tfsi' : '2.0 tfsi komfort',
        'komfort 2.0 tfsi' : '2.0 tfsi komfort',
        '2.0t qtro' : '2.0t',
        'progressive' : 'progressiv',
        "conv":"convertible",
    }
    invalid_audi_trims = ["progressivfinancement a 5.99 60 mois", "komfort gr"]
    words_to_remove = ["tiptronic","s tronic", "quattro"]
    df = process_trim_by_make(df, "audi", audi_trim_correction_mapping, None, words_to_remove, invalid_audi_trims)
    
    # Honda
    invalid_honda_trims = ["lxgarantie 10ans200000km","2 portes"]
    words_to_remove = ["garantie 10 ans 200 000 km","garantie 10 ans200 000 km","navicuirtoit.ouvrant","garantie 10 ans","ouvrant2 cameras",
                      "honda", "ivt","dct"]
    df = process_trim_by_make(df, "honda", None, None, words_to_remove, invalid_honda_trims)
    
    # Ford
    ford_trim_correction_mapping = {
        'xltsport' : 'xlt sport',
        'xlt cabine supercrew 4rm caisse de 6' : "xlt supercrew 6.5ft box",
        'xlt cabine supercrew caisse de 6' : "xlt supercrew 6.5ft box",
        'lariat cabine supercrew caisse de 6':'lariat supercrew 6.5ft box',
        'lariat cabine supercrew caisse de 5':'lariat supercrew 5.5ft box',
        'xlt cabine supercrew caisse de 5':'xlt supercrew 5.5ft box',
        'xlt supercrew 5.5-ft':'xlt supercrew 5.5ft box',
        'eco' : 'ecoboost',
        '150 lightning-lariat cabine supercrew caisse de 5':'150 lightning-lariat',
        '150 lightning-xlt cabine supercrew caisse de 5':'150 lightning-xlt supercrew 5.5ft box',
        '150-lariat cabine supercrew caisse de 5':'150-lariat supercrew 5.5ft box',
        '150-lariat cabine supercrew caisse de 6':'150-lariat',
        '150-xlt cabine supercrew caisse de 5':'supercab 145 xlt',
        '150-xlt cabine supercrew caisse de 6' :'150-xlt supercab 6.5ft box',
        'xlt cabine supercrew caisse de 5 pi':'xlt supercrew 5.5ft box',
        "conv":"convertible", 
        "conv v6":"convertible v6", 
        "conv v6 premium":"convertible v6 premium", 
        "conv gt":"convertible gt", 

        
    }
    valid_ford_trims = ['se', 'xlt', 'sel', 'titanium', 'lariat', 'limited', 'st',
           'gt premium', 'sport', 'platinum', 'xl', 'gt', 'big bend',
           'ecoboost', 'titanium hybrid', 'xlt supercrew 5.5',
           'ecoboost premium', 'outer banks', 'ses', 'raptor', 'badlands',
           'xlt sport', 'king ranch', 'limited max', 'se hatchback']
    df = process_trim_by_make(df, "ford", ford_trim_correction_mapping, valid_ford_trims, None, None)
    
    # Porsche
    words_to_remove = ["pdk", "coupe"]
    df = process_trim_by_make(df, "porsche", None, None, words_to_remove, None)
    
    # Chevrolet
    chevrolet_trim_correction_mapping = {
        'silverado custom' : 'custom',
        'reg cab' : 'regular cab',
        'custom crew' : 'custom crew cab',
        'dbl crew' : 'double cab',
        'crew lt' : 'crew cab lt',
        'cre':'crew cab',
        "conv":"convertible",
    }

    df = process_trim_by_make(df, "chevrolet", chevrolet_trim_correction_mapping, None, words_to_remove, None)
    
    # Chrysler
    chrysler_trim_correction_mapping = {
        'touring l plus' : 'touring-l plus',
        'touring l' : 'touring-l',
        'tourisme' : 'touring',
        '300 s': 's',
        '300 touring' : 'touring',
        '300 c' : 'c',
        'stow nftgo' : 'stow n go',
        "stow’n go" : 'stow n go',
        'sxt stow nftgo' : 'sxt stow n go',
    }

    df = process_trim_by_make(df, make="chrysler", trim_correction_mapping=chrysler_trim_correction_mapping, valid_trims='auto', add_unwanted_words=None, add_invalid_trims=None)
    
    # Dodge
    dodge_trim_correction_mapping = {
        'canada value pkg' : 'canada value package',
        'cvp' : 'canada value package',
        'canada value' : 'canada value package',
        'cvpsxt': 'cvp sxt',
        'groupe valeur canada' : 'canada value package',
        'groupe valeur canada' : 'canada value package',
        '35th anniversary edition' : '35th anniversary',
        '30th anniversary edition' : '30th anniversary',
        "sxt stow’n go" : 'sxt stow n go',
        "sxt stow n'go" : 'sxt stow n go',
        'sxt stow & go' : 'sxt stow n go',
        'sxt stow&go':'sxt stow n go',
        'sxt stowftn go':'sxt stow n go',
        'se stow go':'se stow n go',
        'sxt stow go':'sxt stow n go',
        'stow go':'stow n go',
        'gtawd':'gt',
        'r-t':'rt',
        'crew':'crew cab',
    }

    words_to_remove = ["wgn","cam. de", "i"]

    df = process_trim_by_make(df, make='dodge', trim_correction_mapping=dodge_trim_correction_mapping, valid_trims="auto", add_unwanted_words=words_to_remove, add_invalid_trims=None)
    
    # Hyundai
    hyundai_trim_correction_mapping = {
        'essentiel' : 'essential',
        'cvp' : 'canada value package',
        'pref':'preferred',
        'prefered':'preferred',
        'preffered':'preferred',
        'man l':'l',
        'man gl':'gl',
        'man gls':'gls',
        'preferred 2.0l' : '2.0l preferred',
        'preferred 2.0t' : '2.0t preferred',
        'preferred 2.4' : '2.4l preferred',
        'lux' :'luxury',
        'preferred electric' : 'preferred ev',
    }
    words_to_remove = ["ivt", "dct"]
    
    df = process_trim_by_make(df, make='hyundai', trim_correction_mapping=hyundai_trim_correction_mapping, valid_trims="auto", add_unwanted_words=words_to_remove, add_invalid_trims=None)
    
    # Land Rover
    landrover_trim_correction_mapping = {
        'hse lux' : 'hse luxury',
        'lux' :'luxury'
    }
    df = process_trim_by_make(df, make='land rover', trim_correction_mapping=landrover_trim_correction_mapping, valid_trims="auto", add_unwanted_words=None, add_invalid_trims=None)
    
    # Tesla
    tesla_trim_correction_mapping = {
        'rwd' : 'standard range',
        'standard plus' : 'standard range plus',
        'long range i awd' : 'long range dual motor',
        'long range dual motor awd' : 'long range dual motor',
        'long range awd full self drive' : 'long range dual motor autopilot',
        'long range full self drive' : 'long range autopilot',
        'long range battery' : 'long range',
        'standard range plus full self drive' : 'standard range plus autopilot',
        'standard range plus pilot' : 'standard range plus autopilot',
        'longue autonomie ti' : 'long range autopilot',
        'longue autonomie' : 'long range autopilot',
        'autonomie standard plus' : 'standard range plus autopilot',
        'autonomie standard plus pa' :'standard range plus autopilot',
        'standard range plus autopi' :'standard range plus autopilot',
        'long range 500km autonomie 500k':'long range autopilot',
        'performance full self drive' : 'performance autopilot',
        'long range ltd avail' :'long range -ltd avail',
        'standard range ltd avail' :'standard range -ltd avail',
        'standard':'standard range',
        'dual motor long range':'long range dual motor',
        'dual motor standard range':'standard range dual motor',
        'long range autonome':'long range autopilot',
        'autonome standard plus':'standard range plus',
        'autonome standard plus pa':'standard range plus',
        'longue autonome':'long range autopilot',
        'longue autonome t':'long range autopilot',
        'sr':'standard range',
        'pilot' : 'autopilot',
        'dual motor' : 'long range dual motor',
        'de performance' : 'performance dual motor',
        "standard range plus w autopilot" : "standard range plus autopilot",
        'long range autonomie':"long range autopilot",
        "base":"standard range",
        'long range dual motors' : 'long range dual motor',
        'long range pilot' : 'long range autopilot',
        'autopilot standard plus' : "standard range plus autopilot",
        "autopilot standard plus pa" : "standard range plus autopilot",
        'longue autopilot' : 'long range autopilot',
    }

    invalid_tesla_trims = ["model 3","model s","model y","x","s","3", "360", "electric",'elect',"lo"]

    words_to_remove = ["i","over 30 teslas in stock","and ready", "wow", "sky view", "glass","glassroof","ti","de", "electric"]

    df = process_trim_by_make(df, make='tesla', trim_correction_mapping=tesla_trim_correction_mapping, valid_trims="auto", add_unwanted_words=words_to_remove, add_invalid_trims=invalid_tesla_trims)
    
    # Nissan
    words_to_remove = ["chauffantsbluetooth","svmagcamérabancs", "svawdmagcamérabancs","électriquecamérabluetooth", "awdmagcamérabancs", 
                   "acgr", "awdtoit", "360gpscuir", "slawdcamera 360cuirtoit panomags", "bluetoothregul.vitesseac"]

    df = process_trim_by_make(df, make='nissan', trim_correction_mapping=None, valid_trims="auto", add_unwanted_words=words_to_remove, add_invalid_trims=None)
    
    # Kia
    words_to_remove = ["jamais accidentégarantie 10 ans200 000km","jamais accidentégarantie 10 ans200 000km", "apple car playgarantie 10ans200 000km", "garantie 10 ans", 
                       "ivt","at","man","air","bm","ba","gr"]

    df = process_trim_by_make(df, make='kia', trim_correction_mapping=None, valid_trims="auto", add_unwanted_words=words_to_remove, add_invalid_trims=None)
    
    # Ram 
    ram_trim_correction_mapping = {
        'crew' : 'crew cab',
        'crewcab':' crew cab',
        'cre':'crew cab',
        'cabine quad' : 'quad cab 140.5 st',
        'quad cab 140.5" st' : 'quad cab 140.5 st',
        'laramie crew' : 'laramie crew cab',
        'rebel crew 5ft7inch box' : 'rebel crew 5ft7 box',
        '1500':'base',
        'bighorn':'big horn',
        'crew 140.5inch big horn':'crew 140.5 big horn',
        'crew 140.5inch sport':'crew 140.5 sport',
        'crew 140.5inch st':'crew 140.5 st',
        'express crew 5ft7inch box':'express crew 5ft7 box',
        'express cabine dftéquipe caisse de 5 pi 7 po':'express crew 5ft7 box',
        'express crew bte std . night edi':'express night',
        'longhorn limited':'limited longhorn',
        'ltd':'limited',
        'reg':'base',
        'regular':'base',
    }

    df = process_trim_by_make(df, make='ram', trim_correction_mapping=ram_trim_correction_mapping, valid_trims="auto", add_unwanted_words=None, add_invalid_trims=None, unknown_threshold=4)
    
    # Subaru
    subaru_trim_correction_mapping = {
        '2.5i w-touring pkg' : '2.5i touring',
        '2.5i touring package' : '2.5i touring',
        '3.6r limited package' : '3.6r limited',
        'tourisme' :'touring',
        '2.5i tourisme' :'2.5i touring',
    }
    words_to_remove = ["bm","man","at","mag","man","super","weyesight","w-eyesight","eyesight","w-"]
    df = process_trim_by_make(df, make="subaru", trim_correction_mapping=subaru_trim_correction_mapping, valid_trims="auto", add_unwanted_words=words_to_remove, add_invalid_trims=None)
    
    # Mini
    mini_trim_correction_mapping = {
        'jcw' : 'john cooper works',
        'jc' : 'john cooper works',
        'cooper s' : 's',
        's model' : 's',
        'cooper':'base',
        'hardtop':'base',
        'cooper se':'se',
        'cooper s all4' :'s',
        'all4 s': 's',
        'cooper' : 'base',
        '3 door' : 'base',
        '3-door' : 'base',
        '5 door' :'base',
        '5-door' :'base',
        '5-door' :'base',
        'classic' : 'base classic line',
        'base classic' : 'base classic line',
        'base premier' : 'base premier line',
        'premier' : 'base premier line',
        'premier' : 'base premier line',
    }
    words_to_remove = ["super","all4"]

    df = process_trim_by_make(df, make="mini", trim_correction_mapping=mini_trim_correction_mapping, valid_trims="auto", add_unwanted_words=words_to_remove, add_invalid_trims=None)
    
    # Cadillac
    cadillac_trim_correction_mapping = {
        'luxury 2.0t' : '2.0t luxury',
        'luxury 2.0l' : '2.0l luxury',
        'luxe' : 'luxury',
    }
    words_to_remove = ["collection"]
    
    df = process_trim_by_make(df, make="cadillac", trim_correction_mapping=cadillac_trim_correction_mapping, valid_trims="auto", add_unwanted_words=words_to_remove, add_invalid_trims=None)
    
    # Infiniti
    infiniti_trim_correction_mapping = {
        'technology' : 'tech',
        'technologie':'tech',
        'sport tech' : 'sport tech',
        'sport-tech' : 'sport tech',
        'premium-tech' : 'premium tech',
        'premium tech' : 'premium tech',
        'essentiel' : 'essential',
    }
    words_to_remove = ["proassist"]
    invalid_infiniti_trims = ["qx60","sun","ti","i","utility","safety"]
    df = process_trim_by_make(df, 'infiniti', infiniti_trim_correction_mapping, "auto", words_to_remove, invalid_infiniti_trims)
    
    # Mercedes-Benz
    mercedes_trim_correction_mapping = {
        "c 300": "c300",
        "c 300 4 matic": "c300",
        "c 300 coupe": "c300 coupe",
        "c 300 wagon": "c300 wagon",
        "c 300 amg": "c300 amg",
        "c 450 amg": "c450 amg",
        "c 400": "c400",
        "c 350": "c350",
        "c 250": "c250",
        "amg c 43": "c43 amg",
        "c 43 amg": "c43 amg",
        "c 63 amg": "c63 amg",
        "glc 300": "glc300",
        "glc 300 premium": "glc300 premium",
        "glc 300 coupe": "glc300 coupe",
        "glc 300 amg": "glc300 amg",
        "glc 350e": "glc350e",
        "amg glc 43": "amg glc 43",
        "gle 400": "gle400",
        "gle 450": "gle450",
        "gle 350": "gle350",
        "gle 350d":"gle350d",
        "cla 250": "cla250",
        "cla 250 amg": "cla250 amg",
        "cla 250 coupe" : "cla250 coupe",
        "gla 250": "gla250",
        "gla 250 4 matic" :"gla250",
        "gla 45 amg" :"gla45 amg",
        "e 300": "e300",
        "e 400": "e400",
        "e 350": "e350",
        "e 250 bluetec": "e250 bluetec",
        "e 550": "e550",
        "e 450": "e450",
        "e 400 coupe": "e400 coupe",
        "e 53 amg": "e53 amg",
        "a 250": "a250",
        "a 250 premium" : "a250 premium",
        "a 220": "a220",
        "b 250 routière sport" : "b250 sports tourer",
        "b 250 sports tourer" : "b250 sports tourer",
        "b 250": "b250",
        "s 550": "s550",
        "s 550 lwb": "s550 lwb",
        "s 560": "s560",
        "s 580": "s580",
        "amg s 63" :"s63 amg",
        "glk 250 bluetec": "glk250 bluetec",
        "glk 350": "glk350",
        "gls 450": "gls450",
        "gls 550":"gls550",
        "ml 350 bluetec": "ml350 bluetec",
        "ml 350" : "ml350",
        "g 550" : "g550",
        "c 300" : "c300", 
        "c 300 sport" : "c300 sport", 
        "c 43 amg" : "c43 amg", 
        "c 43 amg premium" : "c43 amg premium", 
        "cla 45 amg" : "cla45 amg", 
        "cls 550" : "cls550", 
        "e 400" : "e400", 
        "g 63 amg" : "g63 amg", 
        "gl 350 bluetec" : "gl350 bluetec", 
        "glb 250" : "glb250", 
        "glc 43 amg" : "glc43 amg", 
        "gle 450 amg" : "gle450 amg", 
        "gle 53" : "gle53", 
        "gls 450 amg" : "gls450 amg", 
        "ml 63 amg" : "ml63 amg", 
        "amg g 63" :"g63 amg",
        "amg a 35" : "amg a35", 
        "amg c 43" : "amg c43", 
        "amg c 43 coupe" : "amg c43 coupe", 
        "amg c43" : "amg c43", 
        "amg cls 53" : "amg cls53", 
        "amg cls 53 coupe" : "amg cls53 coupe", 
        "amg cls 63 s" : "amg cls63 s", 
        "amg e 43" : "amg e43", 
        "amg e 53" : "amg e53", 
        "amg gla 35" : "amg gla35", 
        "amg gla 45" : "amg gla45", 
        "amg glc 43" : "amg glc43", 
        "amg glc 43 coupe" : "amg glc43 coupe", 
        "amg glc 63 s" : "amg glc63 s", 
        "amg gle 43" : "amg gle43", 
        "amg gle 43 coupe" : "amg gle43 coupe", 
        "amg gle 53" : "amg gle53", 
        "amg gle 63 s" : "amg gle63 s", 
        "amg gls 63" : "amg gls63", 
    }
    words_to_remove = ["4matic", "4 matic"]
    df = process_trim_by_make(df, make='mercedes-benz', trim_correction_mapping=mercedes_trim_correction_mapping, valid_trims="auto", add_unwanted_words=words_to_remove, add_invalid_trims=None)
    
    # GMC
    gmc_trim_correction_mapping = {
        'reg' : 'regular',
        'cre':'crew cab',
    }
    
    df = process_trim_by_make(df, make='gmc', trim_correction_mapping=gmc_trim_correction_mapping, valid_trims="auto", add_unwanted_words=None, add_invalid_trims=None)
    
    # Volkswagen
    words_to_remove = ["w heat sts","b. spo","4 motion","4motion"]
    df = process_trim_by_make(df, make='volkswagen', trim_correction_mapping=None, valid_trims="auto", add_unwanted_words=words_to_remove, add_invalid_trims=None)
    
    # Mazda
    mazda_trim_correction_mapping = {
        'gt wturbo' : 'gt turbo',
        'gt w-turbo' : 'gt turbo',
        "gs-sky" : "gs",
        "sport gs-sky" : "sport gs",
        "gt-sky" : "gt",
        "gx-sky" : "gx",
        'gtawd' : 'gt',
        "conv gt":"gt convertible", 
    }
    words_to_remove = ["air","at","sky"]
    df = process_trim_by_make(df, make='mazda', trim_correction_mapping=mazda_trim_correction_mapping, valid_trims="auto", add_unwanted_words=words_to_remove, add_invalid_trims=None)

    # Mitsubishi
    words_to_remove = ["s-awc","awc"]
    df = process_trim_by_make(df, make='mitsubishi', trim_correction_mapping=None, valid_trims="auto", add_unwanted_words=words_to_remove, add_invalid_trims=None)
    
    # Lexus
    words_to_remove = ["series"]
    df = process_trim_by_make(df, make='lexus', trim_correction_mapping=None, valid_trims="auto", add_unwanted_words=words_to_remove, add_invalid_trims=None)
    

    
    # Other makes
    other_makes = ['jeep', 'lincoln', 'volvo']
    for make in other_makes:
        df = process_trim_by_make(df, make, None, 'auto', None, None)
        
    print("Trims of all makes processed")
        
    # Drop the backup trim column
    df = df.drop("trim_backup",axis=1)
    
    # Fill na with unknown
    df['trim'] = df['trim'].fillna("unknown")

    # We will use trims that recur in data at least n times to avoid overfitting
    popular_trims = set(df['trim'].value_counts()[df['trim'].value_counts() >= min_occurrences].index)
    df.loc[~df['trim'].isin(popular_trims), 'trim'] = 'unknown'
    
    # As the same trim names can be found in different makes and this can reduce the performance of ML algorithms, 
    # we combine the trim name with the model name
    if combine_with_modelname:
        df['trim'] = df['model'] + "-" + df['trim']
        df.loc[df['trim'] == 'unknown', 'trim'] = df['model'] + "-unknown"
        
    # Print unique number of trims
    print(f"Number of unique trims: {df.trim.nunique()}")
    
    logging.info(f"SUCCESS: Trim column processed successfully")

    return df