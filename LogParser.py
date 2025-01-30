import pandas as pd
import xml.etree.ElementTree as ET
import json
from abc import ABC, abstractmethod

    # Insieme di classi caratteristiche di un design pattern, dove la classe astratta LogParser definisce un'interfaccia 
    # per il parsing di file di log in diversi formati.
    # Le classi concrete XMLLogParser, CSVLogParser, JSONLogParser, TXTLogParser e XLSXLogParser implementano 
    # il metodo astratto parse per il parsing di file XML, CSV, JSON, TXT e XLSX rispettivamente. 
    # La classe LogParserFactory è una factory che restituisce un parser di file di log in base all'estensione del file. 
    # Questo design pattern è utile per aggiungere nuovi parser di file di log senza dover modificare il codice esistente, 
    # in quanto è sufficiente creare una nuova classe che estende LogParser e implementa il metodo parse.
    #  Inoltre, il codice è più pulito e modulare, in quanto ogni classe ha una responsabilità ben definita e può essere 
    # facilmente sostituita o estesa senza influenzare le altre parti del sistema. 


class LogParser(ABC):
    '''
    Classe astratta per il parsing di file di log in diversi formati.
    '''
    @abstractmethod
    def parse(self, filename: str) -> pd.DataFrame:
        pass

class XMLLogParser:
    '''
    Parser per file XML.
    :param filename: nome del file XML
    :return: pandas DataFrame
    '''
    def parse(self, filename: str) -> pd.DataFrame:
        tree = ET.parse(self.filename)
        root = tree.getroot()
        data = []
        for child in root:
            data.append(child.attrib)
        return pd.DataFrame(data)
    
class CSVLogParser:
    '''
    Parser per file CSV.
    :param filename: nome del file CSV
    :return: pandas DataFrame
    '''
    def parse(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(filename)
    
class JSONLogParser:
    '''
    Parser per file JSON.
    :param filename: nome del file JSON
    :return: pandas DataFrame
    '''
    def parse(self, filename: str) -> pd.DataFrame:
        with open(filename) as f:
            data = json.load(f)
        return pd.DataFrame(data)
    
class TXTLogParser:
    '''
    Parser per file TXT.
    :param filename: nome del file TXT
    :return: pandas DataFrame
    '''
    def parse(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(filename, sep="\t")

class XLSXLogParser:
    '''
    Parser per file XLSX.
    :param filename: nome del file XLSX
    :return: pandas DataFrame
    '''
    def parse(self, filename: str) -> pd.DataFrame:
        return pd.read_excel(filename)
    
class LogParserFactory:
    '''
    Factory per la creazione di parser di file di log in base all'estensione del file
    :param filename: nome del file di log
    :return: parser di file di log
    '''
    def create(self, filename: str) -> LogParser:
        if filename.endswith(".xml"):
            return XMLLogParser()
        elif filename.endswith(".csv"):
            return CSVLogParser()
        elif filename.endswith(".json"):
            return JSONLogParser()
        elif filename.endswith(".txt"):
            return TXTLogParser()
        elif filename.endswith(".xlsx"):
            return XLSXLogParser()
        else:
            raise ValueError("Formato del file non supportato")