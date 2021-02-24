# Utilising Flow Aggregation to Classify Benign Imitating Attacks 


This work aims to introduce new features based on a higher level of abstraction of network traffic, called Flow Aggregation features.
In this repository, two features are extracted: Number of flows, Source ports delta.

## Citation
To be updated.



## General Parameters

| Argument       | Usage        				 	     | Default       |  Values and Notes	          |
| ---------------|:-------------------------------------:|:-------------:|:-------------------|
| --normal_path      | CICIDS2017 Dataset Monday File Path     | csv_files/biflow_Monday-WorkingHours_Fixed_Hour_0.csv  |  |
| --attack_paths     | CICIDS2017 Dataset Attack Paths     | csv_files/biflow_Wednesday-WorkingHours_Slowhttptest.csv  | Comma separated |
| --drop_aggregation | Whether or not to drop flow aggregation features | 0 | 0/1 |
| --slice_normal | Whether or not to use a portion of normal file | 0 | 0/1 |
| --slice_attacks | Whether or not to use portion(s) of attack file(s) | 0 | 0/1 (comma separated) |
| --slice_normal_percent | Percentage of the portion to use from normal file | 0 |  |
| --slice_attacks_percent | Percentage of the portion(s) to use from attack file(s) | 0 | Comma separated |
| --normal_slice_no | Slice number to use when slicing the normal file | 0 |  |
| --slice_attacks_number | Slice number(s) to use when slicing the attack files | 0 | Comma separated |
| --slice_attacks_number | Slice number(s) to use when slicing the attack files | 0 | Comma separated |
| --number_of_features | Number of RFE features to print | 5 | Only used in print_features script |
| --output  | The output file name | result.csv ||
| --choose_features |  Whether or not to run RFE | 0 | 0/1 (check selected_features argument) |
| --selected_features | The features to use during training | 'fwd_mean_pkt_len, bwd_mean_pkt_len, fwd_min_pkt_len, bwd_min_pkt_len, fwd_max_pkt_len,num_src_flows, src_ip_dst_prt_delta' | |



## How to Run the repository:

```
Clone this repository.
run pcap_parser.py [pcap file path] 
run print_features [specify the parameters as required] 
run flow_aggregation.py [specify the parameters as required]

```

- The output of pcap_parser will be saved as 'pcap_file_name.csv'.
- Print features will display the RFE ranked features in order (slicing in parameters is only used if set, default is to use the whole files). 
