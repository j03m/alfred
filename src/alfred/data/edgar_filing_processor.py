from lxml import etree
from collections import defaultdict
import re
from dateutil.parser import parse as parse_date
from alfred.data import OpenFigiDownloader
import pandas as pd
from datetime import datetime

class EdgarFilingProcessor:
    def __init__(self, edgar_database):
        self.open_figi_downloader = OpenFigiDownloader()
        self.xml_parser = etree.XMLParser(recover=True)
        self.edgar_database = edgar_database


    def remove_namespaces(self, xml_content):
        # Remove namespace declarations
        xml_content = re.sub(r'\sxmlns.+(?=>)', '', xml_content, count=0)
        # Remove prefixes in tags
        xml_content = re.sub(r'(<\/?)[\w\d]+:', r'\1', xml_content)
        return xml_content


    def extract_embedded_xml(self, content):
        # Encountering issues because the SEC filing content is not a well-formed XML document—it’s a mix of SGML-like
        # tags and embedded XML sections. Specifically, the presence of an <?xml ... ?> declaration inside the <XML> tags
        # violates XML’s well-formedness rules, causing parsing errors.
        if content is None:
            return {}
        try:
            xml_sections = re.findall(r'<XML>(.*?)</XML>', content, re.DOTALL)
        except Exception as e:
            print("something wrong with content object", e)
            return {}

        roots = {}
        for section in xml_sections:
            # Remove any embedded XML declarations
            xml_string = re.sub(r'<\?xml.*?\?>', '', section)

            # Remove namespaces to simplify XPath expressions down the road
            xml_string = self.remove_namespaces(xml_string)

            # Parse the XML string
            subroot = etree.fromstring(xml_string, parser=self.xml_parser)

            # Get the root tag name
            root_tag_name = subroot.tag

            # Store the parsed XML tree in the dictionary with the root tag name as the key
            roots[root_tag_name] = subroot

        return roots


    def extract_filing_date(self, root):
        # Try to get <dateFiled> first
        date_filed = root.findtext("dateFiled")
        if date_filed:
            date_obj = parse_date(date_filed)
            return date_obj.year, date_obj.month

        # Fallback to <signatureDate> from <signatureBlock>
        signature_block = root.find(".//signatureBlock")
        if signature_block is not None:
            signature_date = signature_block.findtext("signatureDate")
            if signature_date:
                try:
                    date_obj = parse_date(signature_date)
                except Exception as e:
                    print("Failed to parse data: ", e)
                    return None, None
                return date_obj.year, date_obj.month

        raise Exception("Could not find date in dateFiled or signatureBlock")


    def processor(self, content):
        if content is None:
            return

        roots = self.extract_embedded_xml(content)

        if roots.get('edgarSubmission', None) is None:
            return

        year, month = self.extract_filing_date(roots['edgarSubmission'])
        if year is None or month is None:
            return

        table = roots.get("informationTable", None)
        if table is None:
            print(f"Does not contain informationTable")
            return

        for info_table in table.findall(".//infoTable"):
            cusip = info_table.findtext("cusip")
            company_name = info_table.findtext("nameOfIssuer")
            shares = float(info_table.find("shrsOrPrnAmt").findtext("sshPrnamt"))
            try:
                ticker = self.open_figi_downloader.get_ticker_for_cusip(cusip)
            except Exception as e:
                print("Failed hitting open figi:", e, " treating as non fatal")
                ticker = None

            if ticker is None or ticker == "Unknown":
                print(f"Could not find ticker for company: {cusip}/{company_name}")
                continue
            else:
                print(f"{cusip}/{company_name} maps to {ticker} for {shares} shares")

            # Update the DataFrame with month/year, ticker, shares
            self.edgar_database.update(year, month, ticker, shares)

        self.open_figi_downloader.dump_cache()


    def dump_csv(self):
        rows = [
            (datetime(year, month, 1), ticker, value)
            for (year, month, ticker), value in self.aggregated_results.items()
        ]

        df = pd.DataFrame(rows, columns=["date", "ticker", "value"])

        df.to_csv(str(self.output_file))