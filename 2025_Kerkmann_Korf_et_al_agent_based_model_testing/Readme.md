# Information

- **DOI:** [10.48550/arXiv.2410.08050](https://doi.org/10.48550/arXiv.2410.08050)
- **Authors:**  
  David Kerkmann, Sascha Korf, Khoa Nguyen, Daniel Abele, Alain Schengen,  
  Carlotta Gerstein, Jens Henrik Göbbert, Achim Basermann,  
  Martin J. Kühn, Michael Meyer-Hermann

---

# Setup

1. **Download the data file:**  
   [braunschweig_result_ffa8.csv](https://zenodo.org/records/13318436)

2. **Save the file to:**  
   `/memilio/data/mobility/braunschweig_result_ffa8.csv`

3. **Run the data cleanup script:**  
   - Install dependencies:  
     `pip install numpy pandas`
   - Edit `cleanup_data.py` to set correct folder paths.
   - Run the script.

4. **Download simulation files:**  
   Follow the instructions at:  
   [memilio-epidata GitHub](https://github.com/SciCompMod/memilio/tree/main/pycode/memilio-epidata)

5. **Run simulation data extraction:**  

   ```sh
   python getSimulationData.py -s 2021-01-01 -e 2021-07-01 -m 1
   ```

6. **(Optional) Run with reporting date:**  

   ```sh
   python getSimulationData.py -s 2021-01-01 -e 2021-07-01 -m 1 --rep-date
   ```

7. **Copy the Germany folder:**  
   Place it into:  
   `data/mobility/Germany`

8. **Update folder paths:**  
   In `paper_abm_testing`, set the data folder path as needed.

9. **Build the project:**  

   ```sh
   cmake --build /Users/saschakorf/Documents/Arbeit.nosynch/memilio/memilio/cpp/build --config Release --target paper_abm_bs_testing -j 6 --
   ```

10. **Run the executable:**  

    ```sh
    ./memilio/cpp/build/bin/paper_abm_bs_testing
    ```
