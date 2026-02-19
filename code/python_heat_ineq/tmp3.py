def get_recipe(target_data, archive_data, variables):    
    # Intentem amb una tolerància una mica més alta per evitar el 'TypeError'
    # i assegurem que l'archive_data contingui el model desitjat.    
    try:
        target_data['unit'] = "degC change from avg over 1995~2014"        
        stitches_recipe = stitches.make_recipe(
            target_data, 
            archive_data, 
            tol=0.5,  # Pugem a 0.1 per evitar fallades per falta de coincidència exacta
            N_matches=1, 
            res='day', 
            non_tas_variables=[var for var in variables if var != 'tas']
        )
    except Exception as e:
        print(f"Error: {e}")
        return None
    # Ajust de la longitud de l'últim període
    last_period_length = stitches_recipe['target_end_yr'].values[-1] - stitches_recipe['target_start_yr'].values[-1]
    asy = stitches_recipe['archive_start_yr'].values.copy()
    asy[-1] = stitches_recipe['archive_end_yr'].values[-1] - last_period_length
    stitches_recipe['archive_start_yr'] = asy    
    return stitches_recipe









# 1. Carrega només el primer fitxer de la llista
primer_ds = pangeo.fetch_nc(file_list[0])

# 2. Comprova la resolució en graus
# Extraiem la diferència entre píxels de longitud i latitud
lon_res = primer_ds.lon.diff('lon').mean().values
lat_res = primer_ds.lat.diff('lat').mean().values

print(f"Resolució: {lon_res} x {lat_res} graus")

# 3. Comprova les dimensions totals (per saber quants píxels hi ha)
print(f"Dimensions: {primer_ds.dims}")