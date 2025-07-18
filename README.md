# CrossTALK WDR91 Competition 

Artificial Intelligence Ready CHEmiCal Knowledge-base
Overview
The Artificial Intelligence Ready CHEmiCal Knowledge-base (AIRCHECK) is an open platform developed to share large-scale, publicly accessible chemical activity data. Specifically, AIRCHECK is a collection of datasets generated from large DNA-encoded library (DEL) screenings of various protein targets.

The goal of AIRCHECK is to provide both the academic and industry communities with large, well-annotated chemical activity data to help with the development of artificial intelligence (AI)-based methods for hit discovery. AIRCHECK is supported by the Structural Genomics Consortium (SGC) in collaboration with X-Chem and HitGen.

Chemical Representation
Chemicals can be thought of as graphs: a series of nodes (atoms) and edges (bonds) that define the chemical structure. While methods exist that can take graphs as an input (Graph Neural Networks for instance), one often converts chemical structure into a numerical vector (in a somewhat lossy manner) using a process called "chemical fingerprinting".

It is important to note that AIRCHECK only provides precomputed chemical fingerprints of enriched DEL compounds, while the complete chemical structures will not be provided as they remain the intellectual properties of the data providers. It is a violation of the AIRCHECK terms of agreement to try and determine chemical structure from these fingerprints

There are several approaches to chemical fingerprinting. AIRCHECK uses up to 9 of these. Unless otherwise marked, all fingerprints are hashed/count versions:

ECFP4: Extended connectivity fingerprints with 2048 bits and a radius of 4
ECFP6: Extended connectivity fingerprints with 2048 bits and a radius of 6
FCFP4: Feature connectivity fingerprints with 2048 bits and a radius of 4
FCFP6: Feature connectivity fingerprints with 2048 bits and a radius of 6
MACCS: 166 bit common substructure fingerprint BINARY
RDK: The rdkit graph based fingerprint with 2048 bits and max length of 7 BINARY
AVALON: 2048 bit fingerprint from Avalon
TOPTOR: Topological Torsion fingerprint with 2048 bits
ATOMPAIR: Atom Pair fingerprint with 2048 bits
Not all the datasets in AIRCHECK will have all 9 of these fingerprints: the datasets will be marked with which fingerprints will be available after downloading.

Data Overview
AIRCHECK's primary deliverable is large ML-ready datasets. Here we cover exactly what these datasets provide and how that data can be interpreted.

Currently, we provide the data in a simple, universal text format with corresponding metadata (data dictionary). In future releases, SPARK datafiles will also be included.

File format
Files will be provided as compressed gzip and parquet files, gzip files can be uncompressed with gzip -d file.gz. Files will be between 10-50 GB when uncompressed. Columns are tab (\t) separated.

Data Columns
Each dataset will follow the same general format with the following data fields:

Column Idx	Column Name	        Description
0	        DEL_ID	            The unique ID linked this DEL compound. See DEL ID for details
1	        ID	                A unique numerical ID (built from the DEL_ID
2	        TARGET_ID	        Unique ID associated with the target used in screening
3	        RAW_COUNT	        The raw sequence count (enrichment) observed for this DEL compound
4	        LABEL	            A binary classification (0 or 1) for the compound
5	        BB1_ID	            The ID of the building block from cycle 1 (see BB IDs for details)
6	        BB2_ID	            The ID of the building block from cycle 2
7	        BB3_ID	            The ID of the building block from cycle 3 (NULL if 2 cycle DEL compound)
8 and up	<Fingerprint>	    Fingerprint data, 1 column per fingerprint type. See fingerprint
N-1	        MW	                The molecular weight of the compound
N	        CLogP	            The calculated logP of the compound

Data cleaning/filtering
The provided data undergo minimal filtering to allow for as much user-driven creativity as possible. The only steps taken by AIRCHECK and our partners are as follows:

Removal of matrix binders. These are compounds that have high sequence counts due to binding the matrix used to anchor the target for screening.
Removal of promiscuous binders. These are DEL compounds that have been observed as enriched in many other targets as well and thus might just bind protein matter in general.
Binary vs Continuous
The raw continuous output of the DEL screens is included in RAW_COUNT. This count is already corrected for sequence amplification. Our DEL providers also include binary labels for each compound based on their internal processing pipeline, stored in the LABEL column. A value of 1 corresponds to enriched compound, while a value of 0 means not enriched.

It is possible for inactive compounds to have non-zero RAW_COUNT.

Class Balance
The output of a DEL screening only contains data for compounds that were found to be enriched. Thus, for every enriched compound (based on the LABEL column), 10 additional un-enriched compounds are added to the dataset to provide negative example for machine learning and AI analysis. Every dataset will have roughly a mix of 10% active compounds and 90% inactive. These negatives are selected by the DEL providers. They are not selected completely at random but guided by chemically similarity rules.

Fingerprints
For general ease of use and to avoid decompression on the user end, fingerprints are stored as comma-separated strings. For example, 0,0,0,0,11,0,1,0,3,0,0,0,4,0,5.... This can easily be converted to a numpy array using something such as np.array(fingerprint.split(',')).astype(int) Fingerprint columns will be named after their respective fingerprint type. See chemical representation for details and fingerprint types.

DEL ID
Each DEL compound from the library has a unique alphanumeric ID that represents it. These follow the format L##-#(###)-#(###)-#(###), where the content inside the () are optional digits. For example L03-1-2-3, or L03-235-78-2253. In 2 cycle libraries, the 3rd number is left out: L01-123-45. This ID is not random, and some information about the makeup of the DEL compound is encoding inside.

First, the L## corresponds to a library ID, which is unique to a given reaction scheme. If two DELs share a library ID, they were synthesized using the same reaction.

The 1-4 digit numbers that follow are semi-unique ids that correspond to the building block used in that cycle, where the cycles are in the order 1/A, 2/B and 3/C. So that means L03-235-78-2253 used building block (BB) 235 for cycle 1/A, 78 for cycle 2/B and 2253 for cycle 3/C. These IDs are only unique INSIDE the library. Thus, if I had the DEL ID L03-235-235-47, cycles 1/A and 2/B used the same BB, but a different one for 3/C. However, these numerical BB IDs are NOT UNIQUE OUTSIDE a given library. So if I had IDs L03-235-78-2253 and L04-235-78-2253, these two compounds DO NOT share the same building blocks: BB 235 from L03 and 235 from L04 are not the same BB.

BB IDs
Building blocks (BB) can be used in multiple different libraries. This makes the lack of global uniqueness presented in the DEL ID a problem. To address the weak-uniqueness provided by the DEL IDs encoding for building block information, there is also a column for each of the 3 building blocks to store a unique 3 byte ID that IS unique across all libraries. For example, one may have BB 1AX which is BB 345 in L03 and 567 in L04. Thus, DEL compounds L03-345-78-2253 and L04-567-78-194 share the same BB in the cycle 1/A position.

If you want to collect all compounds across all libraries with the same BB, use the BB_ID , not the numeric number in the DEL_ID.

Multi-library Screening
All the DEL selections collected by AIRCHECK include multiple libraries: for example in any dataset generated by HitGen, 50 unique libraries are screened. These results are merged into a single set. All libraries are screened everytime, so just because a compound from L34 do not appear to ever be enriched does not mean that library was not screened, it means that no compound from that library appear to bind the target.

I could be helpful to separate compound out based on libraries, since all compounds from one library share the same reaction schemes (same core scaffold). However, most DEL-ML is done merging all the libraries screened together into one set.

It is also important to remember that not all libraries are the same size. Some libraries are only 2 cycle, thus only have hundreds of thousands or millions of compound. Others are 3 cycle with over a billion compounds. Some are 3 cycle with only 100 compounds in each cycle, thus only has 1,000,000 compounds. In a future release the size of each library screened may be release, but for now this information has not yet been made public. 


There is some uniqueness of DEL chemical fingerprint data that must be accounted for though: chemical fingerprints tend to be sparse (lots of zeros) and can be heavily correlated (in many fingerprints, having a non-zero value in one element requires that another element also be non-zero).

A common practice in the field is to use these as direct inputs into a Random Forest or SVM for instance, combined with feature selection or reduction. 