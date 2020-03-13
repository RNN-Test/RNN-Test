|-- my_paper
    |-- README.md
    |-- adv outputs
    |   |-- audio
    |   |   |-- 2_state_diff_cell_state_adv_1572945539
    |   |       |-- adv_sample-000000.wav
    |   |       |-- adv_sample-000001.wav
    |   |       |-- adv_sample-000002.wav
    |   |       |-- adv_sample-000003.wav
    |   |       |-- adv_sample-000004.wav
    |   |       |-- adv_sample-000005.wav
    |   |       |-- adv_sample-000006.wav
    |   |       |-- adv_sample-000007.wav
    |   |       |-- adv_sample-000008.wav
    |   |       |-- adv_sample-000009.wav
    |   |       |-- adv_sample-000010.wav
    |   |       |-- adv_sample-000011.wav
    |   |       |-- adv_sample-000012.wav
    |   |       |-- adv_sample-000013.wav
    |   |       |-- adv_sample-000014.wav
    |   |       |-- adv_sample-000015.wav
    |   |       |-- adv_sample-000016.wav
    |   |       |-- adv_sample-000017.wav
    |   |       |-- adv_sample-000018.wav
    |   |       |-- adv_sample-000019.wav
    |   |-- ptb
    |   |   |-- 2_state_diff_hidden_state_adv1
    |   |-- spell checker
    |       |-- 2_state_diff_cell_state_adv1
    |-- figures
    |   |-- RQ2
    |   |   |-- perturbation
    |   |       |-- audio
    |   |       |   |-- sample-000000.eps
    |   |       |   |-- sample-000000_vector.txt
    |   |       |   |-- sample-000001.eps
    |   |       |   |-- sample-000001_vector.txt
    |   |       |   |-- sample-000002.eps
    |   |       |   |-- sample-000002_vector.txt
    |   |       |-- ptb
    |   |       |   |-- step_0.eps
    |   |       |   |-- step_0_vector.txt
    |   |       |   |-- step_1.eps
    |   |       |   |-- step_1_vector.txt
    |   |       |   |-- step_2.eps
    |   |       |   |-- step_2_vector.txt
    |   |       |   |-- step_3.eps
    |   |       |   |-- step_3_vector.txt
    |   |       |-- sp
    |   |           |-- step_0.eps
    |   |           |-- step_0_vector.txt
    |   |           |-- step_1.eps
    |   |           |-- step_1_vector.txt
    |   |           |-- step_2.eps
    |   |           |-- step_2_vector.txt
    |   |-- RQ3
    |       |-- correlation
    |       |   |-- audio
    |       |   |   |-- DXSuccess rate.eps
    |       |   |   |-- DXWER.eps
    |       |   |   |-- cell[0]Success rate.eps
    |       |   |   |-- cell[0]WER.eps
    |       |   |   |-- cell[1]Success rate.eps
    |       |   |   |-- cell[1]WER.eps
    |       |   |   |-- cell[2]Success rate.eps
    |       |   |   |-- cell[2]WER.eps
    |       |   |   |-- cell[3]Success rate.eps
    |       |   |   |-- cell[3]WER.eps
    |       |   |   |-- cell[4]Success rate.eps
    |       |   |   |-- cell[4]WER.eps
    |       |   |   |-- hiddenSuccess rate.eps
    |       |   |   |-- hiddenWER.eps
    |       |   |-- ptb
    |       |   |   |-- DX_pp.eps
    |       |   |   |-- cell[0]pp.eps
    |       |   |   |-- cell[1]pp.eps
    |       |   |   |-- cell[2]pp.eps
    |       |   |   |-- cell[3]pp.eps
    |       |   |   |-- cell[4]pp.eps
    |       |   |   |-- hidden_pp.eps
    |       |   |-- sp
    |       |       |-- DXSuccess rate_sp.eps
    |       |       |-- DXWER_sp.eps
    |       |       |-- cell[0]Success rate_sp.eps
    |       |       |-- cell[0]WER_sp.eps
    |       |       |-- cell[1]Success rate_sp.eps
    |       |       |-- cell[1]WER_sp.eps
    |       |       |-- cell[2]Success rate_sp.eps
    |       |       |-- cell[2]WER_sp.eps
    |       |       |-- cell[3]Success rate_sp.eps
    |       |       |-- cell[3]WER_sp.eps
    |       |       |-- cell[4]Success rate_sp.eps
    |       |       |-- cell[4]WER_sp.eps
    |       |       |-- hiddenSuccess rate_sp.eps
    |       |       |-- hiddenWER_sp.eps
    |       |-- distribution
    |           |-- audio
    |           |   |-- DX_box_toge.eps
    |           |   |-- cell[0]_box_toge.eps
    |           |   |-- cell[1]_box_toge.eps
    |           |   |-- cell[2]_box_toge.eps
    |           |   |-- cell[3]_box_toge.eps
    |           |   |-- cell[4]_box_toge.eps
    |           |   |-- hidden_box_toge.eps
    |           |-- ptb
    |           |   |-- DX_box_toge.eps
    |           |   |-- cell[0]_box_toge.eps
    |           |   |-- cell[1]_box_toge.eps
    |           |   |-- cell[2]_box_toge.eps
    |           |   |-- cell[3]_box_toge.eps
    |           |   |-- cell[4]_box_toge.eps
    |           |   |-- hidden_box_toge.eps
    |           |-- sp
    |               |-- DX_box_toge.eps
    |               |-- cell[0]_box_toge.eps
    |               |-- cell[1]_box_toge.eps
    |               |-- cell[2]_box_toge.eps
    |               |-- cell[3]_box_toge.eps
    |               |-- cell[4]_box_toge.eps
    |               |-- hidden_box_toge.eps
    |-- src
        |-- .idea
        |   |-- misc.xml
        |   |-- modules.xml
        |   |-- src.iml
        |   |-- workspace.xml
        |-- audio
        |   |-- coverage.py
        |   |-- filterbanks.npy
        |   |-- gen_adv_exp.py
        |   |-- make_checkpoint.py
        |   |-- statefulRNNCell.py
        |   |-- tf_logits.py
        |   |-- commonvoice
        |   |   |-- LICENSE
        |   |   |-- README
        |   |   |-- labels.csv
        |   |-- testdata
        |       |-- sample-000000.wav
        |       |-- sample-000001.wav
        |       |-- sample-000002.wav
        |       |-- sample-000003.wav
        |       |-- sample-000004.wav
        |       |-- sample-000005.wav
        |       |-- sample-000006.wav
        |       |-- sample-000007.wav
        |       |-- sample-000008.wav
        |       |-- sample-000009.wav
        |       |-- sample-000010.wav
        |       |-- sample-000011.wav
        |       |-- sample-000012.wav
        |       |-- sample-000013.wav
        |       |-- sample-000014.wav
        |       |-- sample-000015.wav
        |       |-- sample-000016.wav
        |       |-- sample-000017.wav
        |       |-- sample-000018.wav
        |       |-- sample-000019.wav
        |-- ptb
        |   |-- coverage.py
        |   |-- gen_adv_exp.py
        |   |-- myreader.py
        |   |-- ptb_word_lm_wrapped.py
        |   |-- ckpt
        |   |   |-- checkpoint
        |   |   |-- model.ckpt-13.data-00000-of-00001
        |   |   |-- model.ckpt-13.index
        |   |   |-- model.ckpt-13.meta
        |   |-- testdata
        |       |-- ptb.test.txt
        |       |-- ptb.train.txt
        |       |-- ptb.valid.txt
        |-- spell checker
            |-- SpellChecker_wrapped.py
            |-- coverage_sp.py
            |-- gen_adv_exp.py
            |-- kp=0.85,nl=2,th=0.95,11.ckpt.data-00000-of-00001
            |-- kp=0.85,nl=2,th=0.95,11.ckpt.index
            |-- kp=0.85,nl=2,th=0.95,11.ckpt.meta
            |-- last_checkpoints
            |-- statefulRNNCell.py
            |-- test_seq_short
            |-- books
                |-- Alices_Adventures_in_Wonderland_by_Lewis_Carroll.rtf
                |-- Anna_Karenina_by_Leo_Tolstoy.rtf
                |-- David_Copperfield_by_Charles_Dickens.rtf
                |-- Don_Quixote_by_Miguel_de_Cervantes.rtf
                |-- Dracula_by_Bram_Stoker.rtf
                |-- Emma_by_Jane_Austen.rtf
                |-- Frankenstein_by_Mary_Shelley.rtf
                |-- Great_Expectations_by_Charles_Dickens.rtf
                |-- Grimms_Fairy_Tales_by_The_Brothers_Grimm.rtf
                |-- Metamorphosis_by_Franz_Kafka.rtf
                |-- Oliver_Twist_by_Charles_Dickens.rtf
                |-- Pride_and_Prejudice_by_Jane_Austen.rtf
                |-- The_Adventures_of_Sherlock_Holmes_by_Arthur_Conan_Doyle.rtf
                |-- The_Adventures_of_Tom_Sawyer_by_Mark_Twain.rtf
                |-- The_Count_of_Monte_Cristo_by_Alexandre_Dumas.rtf
                |-- The_Picture_of_Dorian_Gray_by_Oscar_Wilde.rtf
                |-- The_Prince_by_Nicolo_Machiavelli.rtf
                |-- The_Romance_of_Lust_by_Anonymous.rtf
                |-- The_Yellow_Wallpaper_by_Charlotte_Perkins_Gilman.rtf
                |-- Through_the_Looking_Glass_by_Lewis_Carroll.rtf
