
from modules.utils import firefox_dataset_p2 as fd


class BR_Feat_Final_Matrix_Handler:
    
    def __get_features(self, br_id, data_origin):
        features_ids = ""
        matrix = ""

        if data_origin == do.DataOrigin.VOLUNTEERS_AND_EXPERT_UNION:
            matrix = exp_vol_union_matrix
        elif data_origin == do.DataOrigin.VOLUNTEERS_AND_EXPERT_INTERSEC:
            matrix = exp_vol_intersec_matrix
        elif data_origin == do.DataOrigin.EXPERT:
            matrix = expert_matrix
        elif data_origin == do.DataOrigin.VOLUNTEERS:
            matrix = volunteers_matrix

        for col in matrix.columns:
            if matrix.at[br_id, col] == 1:
                if features_ids == "":
                    features_ids = str(matrix.columns.get_loc(col) + 1)
                else:
                    features_ids = features_ids + " " + str(matrix.columns.get_loc(col) + 1)

        return features_ids

    
    def compile_results(self, data_origin):
        br_2_features_matrix_final = pd.DataFrame()
        br_2_features_matrix_final['Bug_Number'] = bugreports.Bug_Number
        br_2_features_matrix_final['Features_IDs_exp_m'] = bugreports.apply(lambda row : get_features(row['Bug_Number'], do.DataOrigin.EXPERT), axis=1)
        br_2_features_matrix_final['Features_IDs_vol_m'] = bugreports.apply(lambda row : get_features(row['Bug_Number'], do.DataOrigin.VOLUNTEERS), axis=1)
        br_2_features_matrix_final['Features_IDs_exp_vol_union_m'] = bugreports.apply(lambda row : get_features(row['Bug_Number'], do.DataOrigin.VOLUNTEERS_AND_EXPERT_UNION), axis=1)
        br_2_features_matrix_final['Features_IDs_exp_vol_intersec_m'] = bugreports.apply(lambda row : get_features(row['Bug_Number'], do.DataOrigin.VOLUNTEERS_AND_EXPERT_INTERSEC), axis=1)
        br_2_features_matrix_final.replace(" ", "", inplace=True)
        
        return br_2_features_matrix_final
        
    
    def save_br_feat_final_matrix(self):
        fd.Feat_BR_Oracles.write_br_2_features_matrix_final_df(br_2_features_matrix_final)