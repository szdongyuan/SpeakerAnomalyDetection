#!/usr/bin/env python3
import argparse
import os
import sys
import hashlib
from base.db_manager import DataSave
from base.system_intervction.hardware_intervction import get_mac_address
from consts import model_consts, error_code

def get_db_path(p):
    return p if p else model_consts.DATABASE_PATH

def encrypt_password(user_name, password):
    mac_pwd = get_mac_address() + user_name + password
    sh = hashlib.sha1()
    sh.update(mac_pwd.encode("utf-8"))
    return sh.hexdigest()

def main():
    ap = argparse.ArgumentParser(description="Seed or update Admin user with hashed password")
    ap.add_argument("--db", default=None)
    ap.add_argument("--user", default=os.getenv("ADMIN_USER"))
    ap.add_argument("--password", default=os.getenv("ADMIN_PASS"))
    args = ap.parse_args()

    if not args.user or not args.password:
        print("Missing --user/--password or ADMIN_USER/ADMIN_PASS", file=sys.stderr)
        sys.exit(2)

    enc_pwd = encrypt_password(args.user, args.password)
    db_path = get_db_path(args.db)
    with DataSave(db_path) as db:
        db.create_table()
        code, data = db.query("users_table", ["COUNT(*)"], {"access_level": "Admin"})
        if code != error_code.OK:
            print("Failed to query admin count", file=sys.stderr)
            sys.exit(1)
        count = data[0][0] if data else 0
        if count and int(count) > 0:
            qcode, qdata = db.query("users_table", ["user_name"], {"access_level": "Admin"})
            if qcode != error_code.OK or not qdata:
                print("Failed to query existing admin", file=sys.stderr)
                sys.exit(1)
            current_admin = qdata[0][0]
            ucode, msg = db.update_table_data("users_table", {"password": enc_pwd}, {"user_name": current_admin}, update_time=True)
            if ucode != error_code.OK:
                print(msg, file=sys.stderr)
                sys.exit(1)
            print(f"Admin exists: {current_admin}. Password updated.")
            return
        insert_code, msg = db.insert_data_into_db("users_table", model_consts.USERS_COLUMNS, [(args.user, enc_pwd, "Admin")])
        if insert_code != error_code.OK:
            print(msg, file=sys.stderr)
            sys.exit(1)
        print("Admin user created")

if __name__ == "__main__":
    main()
