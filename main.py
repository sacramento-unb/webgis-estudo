import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static,st_folium
import streamlit as st
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
from rasterio.io import MemoryFile
from folium.raster_layers import ImageOverlay
import cv2
from zona_utm import calcular_utm
from utils import color_map,value_to_class

st.title('Estudo de caso WebGIS')
st.write('')
st.write('')
st.sidebar.title('Menu')

# Upload de um arquivo
poligono_subido = st.sidebar.file_uploader(label='Selecione um polígono a ser analisado')
raster_subido = st.sidebar.file_uploader("Selecione o raster (GeoTIFF)", type=["tif", "tiff"])

# Checagem para saber se o arquivo foi subido

EMBARGO = 'dados/embargos/embargos_ibama.parquet'
DESMATAMENTO = 'dados/mapbiomas/mapbiomas_alertas.parquet'
TIS = 'dados/tis_poligonais/tis.parquet'

if poligono_subido and raster_subido:
    poligono_analise = gpd.read_file(poligono_subido)
    epsg = calcular_utm(poligono_analise)

    gdf = poligono_analise

    @st.cache_resource
    def abrir_embargo():
        gdf_embargo = gpd.read_parquet(EMBARGO)
        return gdf_embargo
    
    @st.cache_resource
    def abrir_desmatamento():
        gdf_desmat = gpd.read_parquet(DESMATAMENTO)
        return gdf_desmat
    
    @st.cache_resource
    def abrir_tis():
        gdf_ti = gpd.read_parquet(TIS)
        return gdf_ti
    
    gdf_embargo = abrir_embargo()

    gdf_desmat = abrir_desmatamento()

    gdf_ti = abrir_tis()

    #st.dataframe(gdf_embargo.head())

    gdf_embargo = gdf_embargo.drop(columns=['nom_pessoa','cpf_cnpj_i',
                            'cpf_cnpj_s','end_pessoa',
                            'des_bairro','num_cep','num_fone',
                            'data_tad','dat_altera','data_cadas',
                            'data_geom','dt_carga'])

    entrada_embargo = gpd.sjoin(gdf_embargo,gdf,how='inner',predicate='intersects')

    entrada_embargo = gpd.overlay(entrada_embargo,gdf,how='intersection')

    entrada_desmat = gpd.sjoin(gdf_desmat,gdf,how='inner',predicate='intersects')

    entrada_desmat = gpd.overlay(entrada_desmat,gdf,how='intersection')

    entrada_ti = gpd.sjoin(gdf_ti,gdf,how='inner',predicate='intersects')

    entrada_ti = gpd.overlay(entrada_ti,gdf,how='intersection')
    #Conversão do geodataframe em um dataframe

    area_desmat = entrada_desmat.dissolve(by=None)

    area_desmat = area_desmat.to_crs(epsg=epsg)

    area_desmat['area'] = area_desmat.area / 10000



    area_embargo = entrada_embargo.dissolve(by=None)

    area_embargo = area_embargo.to_crs(epsg=epsg)

    area_embargo['area'] = area_embargo.area / 10000



    area_ti = entrada_ti.dissolve(by=None)

    area_ti = area_ti.to_crs(epsg=epsg)

    area_ti['area'] = area_ti.area / 10000

    card_columns1,card_columns2,card_columns3 = st.columns(3)

    with card_columns1:
        st.write('Área Total desmatada')
        if len(area_desmat) == 0:
            st.subheader('0')
        else:
            st.subheader(str(round(area_desmat.loc[0,'area'],2)))

    with card_columns2:
        st.write('Área Total de embargos')
        if len(area_embargo) == 0:
            st.subheader('0')
        else:
            st.subheader(str(round(area_embargo.loc[0,'area'],2)))

    with card_columns3:
        st.write('Área Total de Terras Indígenas')
        if len(area_ti) == 0:
            st.subheader('0')
        else:
            st.subheader(str(round(area_ti.loc[0,'area'],2)))

    with MemoryFile(raster_subido.getvalue()) as memfile:
        with memfile.open() as src:
            polygon = gpd.read_file(poligono_subido)

            # Ensure CRS match
            if polygon.crs != src.crs:
                polygon = polygon.to_crs(src.crs)

            # Mask raster with polygon
            geometries = polygon.geometry
            out_image, out_transform = mask(src, geometries, crop=True)
            out_image = out_image[0]  # Use only first band

            # Compute bounds correctly
            height, width = out_image.shape
            min_x, min_y = out_transform * (0, 0)  # Top-left
            max_x, max_y = out_transform * (width, height)  # Bottom-right

            # Compute centroid
            centroid_x, centroid_y = (min_x + max_x) / 2, (min_y + max_y) / 2

            # Convert raster to color image
            rgb_image = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA image

            for value, color in color_map.items():
                rgb_image[out_image == value] = color

            # Resize image for better display
            resized_image = cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_NEAREST)

            # Create interactive map
            m = folium.Map(location=[centroid_y, centroid_x], zoom_start=8, tiles="Esri World Imagery")

            folium.TileLayer(
                tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
                attr='CartoDB Positron',
                name='CartoDB Positron'
            ).add_to(m)

            folium.TileLayer(
                tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
                attr='CartoDB Dark Matter',
                name='CartoDB Dark Matter'
            ).add_to(m)

            # Add OpenTopoMap
            folium.TileLayer(
                tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
                attr='OpenTopoMap',
                name='OpenTopoMap'
            ).add_to(m)

            # Add raster overlay
            bounds = [[min_y, min_x], [max_y, max_x]]
            
            # Add imagem overlays
            ImageOverlay(
                image=resized_image,
                bounds=bounds,
                opacity=0.7,
                name='Mapbiomas Coleção 9',
                interactive=True,
                cross_origin=False,
                zindex=1
            ).add_to(m)


            def style_function_entrada(x): return{
                'fillColor': 'blue',
                'color':'black',
                'weight':0,
                'fillOpacity':0.3
            }

            def style_function_embargo(x): return{
                'fillColor': 'orange',
                'color':'black',
                'weight':1,
                'fillOpacity':0.6
            }

            def style_function_desmat(x): return{
                'fillColor': 'red',
                'color':'black',
                'weight':1,
                'fillOpacity':0.6
            }

            def style_function_ti(x): return{
                'fillColor': 'yellow',
                'color':'black',
                'weight':1,
                'fillOpacity':0.6
            }

            entrada_embargo_limpo = gpd.GeoDataFrame(entrada_embargo,columns=['geometry'])
            folium.GeoJson(entrada_embargo_limpo,name='Embargos IBAMA',style_function=style_function_embargo).add_to(m)
            entrada_desmat_limpo = gpd.GeoDataFrame(entrada_desmat,columns=['geometry'])
            folium.GeoJson(entrada_desmat_limpo,name='Mapbiomas Alertas',style_function=style_function_desmat).add_to(m)
            entrada_ti_limpo = gpd.GeoDataFrame(entrada_ti,columns=['geometry'])
            folium.GeoJson(entrada_ti_limpo,name='TI FUNAI',style_function=style_function_ti).add_to(m)

            folium.LayerControl().add_to(m)

            m.fit_bounds(bounds)

            st_folium(m, width="100%")

            st.write(f"Coordenadas do centroide: ({centroid_x}, {centroid_y})")
            
            # Assuming out_image is your image array
            unique_values, counts = np.unique(out_image, return_counts=True)

            st.write("Áreas em hectares:")
            for value, count in zip(unique_values, counts):
                class_name = value_to_class.get(value, "Unknown")  # Get class name, default to "Unknown" if not found
                area_ha = (count * 900) / 10000
                st.write(f"{class_name}, {area_ha} (ha)")


    df_embargo = pd.DataFrame(entrada_embargo).drop(columns=['geometry'])

    df_desmat = pd.DataFrame(entrada_desmat).drop(columns=['geometry'])

    df_ti = pd.DataFrame(entrada_ti).drop(columns=['geometry'])

    col1_graf,col2_graf,col3_graf,col4_graf = st.columns(4)


    tema_grafico = col1_graf.selectbox('Selecione o tema do gráfico',
                options=['Embargo','Desmatamento','Terras Indígenas'])

    if tema_grafico == 'Embargo':
        df_analisado = df_embargo
    elif tema_grafico == 'Desmatamento':
        df_analisado = df_desmat
    elif tema_grafico == 'Terras Indígenas':
        df_analisado = df_ti

    tipo_grafico = col2_graf.selectbox('Selecione o tipo de gráfico',
                    options=['box','bar','line','scatter','violin','histogram'],index=5)

    plot_func = getattr(px, tipo_grafico)
    # criação de opções dos eixos x e y com um opção padrão
    x_val = col3_graf.selectbox('Selecione o eixo x',options=df_analisado.columns,index=6)

    y_val = col4_graf.selectbox('Selecione o eixo y',options=df_analisado.columns,index=5)
    # Crio a plotagem do gráfico
    plot = plot_func(df_analisado,x=x_val,y=y_val)
    # Faço a plotagem
    st.plotly_chart(plot, use_container_width=True)
    
else:
    st.warning('Selecione os arquivos arquivo para iniciar o WebGIS')